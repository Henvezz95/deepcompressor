# -*- coding: utf-8 -*-
"""Collect calibration dataset."""

import os
from dataclasses import dataclass

import sys
sys.path.append('/workspace/deepcompressor/Infinity_rep/')

import datasets
import torch
from omniconfig import configclass
from torch import nn
from tqdm import tqdm

from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.utils.common import hash_str_to_int, tree_map, tree_split
from Infinity_rep.infinity.models.infinity import Infinity 
from Infinity_rep.tools.run_infinity import *

from deepcompressor.app.diffusion.dataset.data import get_dataset

from Infinity_rep.infinity.models.infinity import Infinity, SelfAttnBlock, CrossAttnBlock, sample_with_top_k_top_p_also_inplace_modifying_logits_
from Infinity_rep.tools.run_infinity import load_visual_tokenizer, load_tokenizer, gen_one_img
from Infinity_rep.infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from deepcompressor.app.diffusion.nn.struct_infinity import patchModel 
from contextlib import contextmanager

import inspect
import typing as tp
import functools

import argparse

model_path = '/workspace/Infinity/weights/infinity_2b_reg.pth'
vae_path = '/workspace/Infinity/weights/infinity_vae_d32reg.pth'
text_encoder_ckpt = '/workspace/Infinity/weights/flan-t5-xl'
h_div_w = 1/1 
enable_positive_prompt = 0

args = argparse.Namespace(
    pn='1M',
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=32,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_2b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    text_channels=2048,
    apply_spatial_patchify=0,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir='/dev/shm',
    checkpoint_type='torch',
    seed=0,
    bf16=1,
    save_file='tmp.jpg',
    enable_model_cache=0
)

h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]


# --- 1. Create a Child Class for Calibration ---
class StatefulInfinity(Infinity):
    """
    An Infinity model subclass designed for stateful data collection.
    It hooks into the autoregressive loop to capture the complete
    state required to replay a forward pass at any given step.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This will store the state for ONE image generation.
        self.kv_delta_cache = {}
        self.caches_dirpath = ''
        self.filename = ''

    def _capture_kv_delta_hook(self, layer_name, module, input, output):
        """The hook function that will be attached to each PatchedSelfAttention layer."""
        # The hook captures the *output* of the K and V projection layers.
        # We assume the patched forward pass computes k and v from hidden_states.
        # This needs to capture the `k` and `v` variables from within the attention forward pass.
        # Let's adjust the PatchedSelfAttention to make this easier.
        # For now, let's assume we can get k and v.
        # In a real implementation, we might need to modify the patched attention to store them.
        k, v = output # This assumes the attention layer returns (out, k, v)
        self.kv_delta_cache[layer_name] = {
            'k_delta': k.detach().cpu(),
            'v_delta': v.detach().cpu()
        }

    @contextmanager
    def _register_delta_hooks(self):
        """A context manager to register and remove hooks automatically."""
        hooks = []
        try:
            for name, module in self.named_modules():
                if isinstance(module, CrossAttnBlock):
                    # We attach the hook to the self-attention submodule
                    sa_layer = module.sa
                    hook = sa_layer.register_forward_hook(
                        lambda mod, inp, outp, n=name: self._capture_kv_delta_hook(f"{n}.sa", mod, inp, outp)
                    )
                    hooks.append(hook)
            yield
        finally:
            for hook in hooks:
                hook.remove()

    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
        g_seed=None, cfg_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,
        returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,
        cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],
        vae_type=0, softmax_merge_topk=-1, ret_img=False,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        inference_mode=False,
        save_img_path=None,
        sampling_per_bits=1,
    ):   
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)

        # scale_schedule is used by infinity, vae_scale_schedule is used by vae if there exists a spatial patchify, 
        # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if any(np.array(cfg_list) != 1):
            bs = 2*B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
        else:
            bs = B

        kv_compact = self.text_norm(kv_compact)
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) # sos shape: [2, 4096]
        kv_compact = self.text_proj_for_ca(kv_compact) # kv_compact shape: [304, 4096]
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)

        with torch.amp.autocast('cuda', enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
        accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
        idx_Bl_list, idx_Bld_list = [], []

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(True)
        
        abs_cfg_insertion_layers = []
        add_cfg_on_logits, add_cfg_on_probs = False, False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            if item == 0: # add cfg on logits
                add_cfg_on_logits = True
            elif item == 1: # add cfg on probs
                add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
            elif item < 0: # determine to add cfg at item-th layer's output
                assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}'
                abs_cfg_insertion_layers.append(leng+item)
            else:
                raise ValueError(f'cfg_insertion_layer: {item} is not valid')
        
        num_stages_minus_1 = len(scale_schedule)-1
        summed_codes = 0
        for si, pn in enumerate(scale_schedule):
            cfg = cfg_list[si]

            if si >= trunk_scale:
                break
            cur_L += np.array(pn).prod()

            need_to_pad = 0
            attn_fn = None
            if self.use_flex_attn:
                # need_to_pad = (self.pad_to_multiplier - cur_L % self.pad_to_multiplier) % self.pad_to_multiplier
                # if need_to_pad:
                #     last_stage = F.pad(last_stage, (0, 0, 0, need_to_pad))
                attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)

            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            layer_idx = 0
            # Reset the temporary delta cache for this step
            self.kv_delta_cache = {}

            # >>> HOOK POINT: The forward pass of the blocks will trigger our hooks <<<
            with self._register_delta_hooks():
                for block_idx, b in enumerate(self.block_chunks):
                    # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
                    if self.add_lvl_embeding_only_first_block and block_idx == 0:
                        last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                    if not self.add_lvl_embeding_only_first_block: 
                        last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                    
                    for m in b.module:
                        last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, scale_ind=si)
                        if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                            # print(f'add cfg={cfg} on {layer_idx}-th layer output')
                            last_stage = cfg * last_stage[:B] + (1-cfg) * last_stage[B:]
                            last_stage = torch.cat((last_stage, last_stage), 0)
                        layer_idx += 1

            current_step_state = {
                'input_hidden_states': last_stage.detach().cpu(),
                'cond_BD': cond_BD_or_gss.detach().cpu(),
                'ca_kv': tree_map(lambda t: t.detach().cpu() if isinstance(t, torch.Tensor) else t, ca_kv),
                'scale_schedule': scale_schedule,
                'scale_ind': si,
                'rope2d_freqs_grid': self.rope2d_freqs_grid,
                'kv_deltas': self.kv_delta_cache, # Save the captured deltas
            }
            
            # Save this single step's state to its own file
            save_path = os.path.join(self.caches_dirpath, f"{self.filename}_step_{si:02d}.pt")
            torch.save(current_step_state, save_path)
            
            if (cfg != 1) and add_cfg_on_logits:
                # print(f'add cfg on add_cfg_on_logits')
                logits_BlV = self.get_logits(last_stage, cond_BD).mul(1/tau_list[si])
                logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
            else:
                logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
            
            if self.use_bit_label:
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
            else:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
            
            if vae_type != 0:
                assert returns_vemb
                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[si]
                else:
                    assert pn[0] == 1
                    idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1) # shape: [B, h, w, d] or [B, h, w, 4d]
                    if self.apply_spatial_patchify: # unpatchify operation
                        idx_Bld = idx_Bld.permute(0,3,1,2) # [B, 4d, h, w]
                        idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, d, 2h, 2w]
                        idx_Bld = idx_Bld.permute(0,2,3,1) # [B, 2h, 2w, d]
                    idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

                idx_Bld_list.append(idx_Bld)
                codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                if si != num_stages_minus_1:
                    summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                    last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
                    if self.apply_spatial_patchify: # patchify operation
                        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
                    last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                    last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
                else:
                    summed_codes += codes
            else:
                if si < gt_leak:
                    idx_Bl = gt_ls_Bl[si]
                h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl)  # BlC

                # h_BChw = h_BChw.float().transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1])
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1], scale_schedule[si][2])
                ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
                idx_Bl_list.append(idx_Bl)
                if si != num_stages_minus_1:
                    accu_BChw, last_stage = self.quant_only_used_in_inference[0].one_step_fuse(si, num_stages_minus_1+1, accu_BChw, h_BChw, scale_schedule)
            
            if si != num_stages_minus_1:
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs//B, 1, 1)

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(False)

        if not ret_img:
            return ret, idx_Bl_list, []
        
        if vae_type != 0:
            img = vae.decode(summed_codes.squeeze(-3))
        else:
            img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)

        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
        return ret, idx_Bl_list, img
    

def load_infinity(
    rope2d_each_sa_layer, 
    rope2d_normalized_by_hw, 
    use_scale_schedule_embedding, 
    pn, 
    use_bit_label, 
    add_lvl_embeding_only_first_block, 
    model_path='', 
    scale_schedule=None, 
    vae=None, 
    device='cuda', 
    model_kwargs=None,
    text_channels=2048,
    apply_spatial_patchify=0,
    use_flex_attn=False,
    bf16=False,
    checkpoint_type='torch',
):
    print(f'[Loading Infinity]')
    text_maxlen = 512
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        infinity_test: StatefulInfinity = StatefulInfinity(
            vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
            shared_aln=True, raw_scale_schedule=scale_schedule,
            checkpointing='full-block',
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=use_flex_attn,
            add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label,
            rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pn=pn,
            apply_spatial_patchify=apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            **model_kwargs,
        ).to(device=device)
        print(f'[you selected Infinity with {model_kwargs=}] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')

        if bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        infinity_test.eval()
        infinity_test.requires_grad_(False)

        infinity_test.cuda()
        torch.cuda.empty_cache()

        print(f'[Load Infinity weights]')
        if checkpoint_type == 'torch':
            state_dict = torch.load(model_path, map_location=device)
            print(infinity_test.load_state_dict(state_dict))
        elif checkpoint_type == 'torch_shard':
            from transformers.modeling_utils import load_sharded_checkpoint
            load_sharded_checkpoint(infinity_test, model_path, strict=False)
        infinity_test.rng = torch.Generator(device=device)
        return infinity_test

def load_transformer(vae, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    if args.checkpoint_type == 'torch': 
        # copy large model to local; save slim to local; and copy slim to nas; load local slim model
        if osp.exists(args.cache_dir):
            local_model_path = osp.join(args.cache_dir, 'tmp', model_path.replace('/', '_'))
        else:
            local_model_path = model_path
        if args.enable_model_cache:
            slim_model_path = model_path.replace('ar-', 'slim-')
            local_slim_model_path = local_model_path.replace('ar-', 'slim-')
            os.makedirs(osp.dirname(local_slim_model_path), exist_ok=True)
            print(f'model_path: {model_path}, slim_model_path: {slim_model_path}')
            print(f'local_model_path: {local_model_path}, local_slim_model_path: {local_slim_model_path}')
            if not osp.exists(local_slim_model_path):
                if osp.exists(slim_model_path):
                    print(f'copy {slim_model_path} to {local_slim_model_path}')
                    shutil.copyfile(slim_model_path, local_slim_model_path)
                else:
                    if not osp.exists(local_model_path):
                        print(f'copy {model_path} to {local_model_path}')
                        shutil.copyfile(model_path, local_model_path)
                    save_slim_model(local_model_path, save_file=local_slim_model_path, device=device)
                    print(f'copy {local_slim_model_path} to {slim_model_path}')
                    if not osp.exists(slim_model_path):
                        shutil.copyfile(local_slim_model_path, slim_model_path)
                        os.remove(local_model_path)
                        os.remove(model_path)
            slim_model_path = local_slim_model_path
        else:
            slim_model_path = model_path
        print(f'load checkpoint from {slim_model_path}')
    elif args.checkpoint_type == 'torch_shard':
        slim_model_path = model_path

    if args.model_type == 'infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8) # 2b model
    elif args.model_type == 'infinity_8b':
        kwargs_model = dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    elif args.model_type == 'infinity_layer12':
        kwargs_model = dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer16':
        kwargs_model = dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer24':
        kwargs_model = dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer32':
        kwargs_model = dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer40':
        kwargs_model = dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer48':
        kwargs_model = dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    infinity = load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer, 
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=args.use_scale_schedule_embedding,
        pn=args.pn,
        use_bit_label=args.use_bit_label, 
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block, 
        model_path=slim_model_path, 
        scale_schedule=None, 
        vae=vae, 
        device=device, 
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
        checkpoint_type=args.checkpoint_type,
    )
    return infinity

# LOAD
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
vae = load_visual_tokenizer(args)

def process(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    return torch.from_numpy(x.float().numpy()).to(dtype)


def create_stateful_cache(config: DiffusionPtqRunConfig, dataset: datasets.Dataset):
    samples_dirpath = os.path.join(config.output.root, "samples")
    caches_dirpath = os.path.join(config.output.root, "caches")
    os.makedirs(samples_dirpath, exist_ok=True)
    os.makedirs(caches_dirpath, exist_ok=True)

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    stateful_model = load_transformer(vae, args)
    model = patchModel(stateful_model) # Use the patching function from your struct file

    batch_size = config.eval.batch_size
    print(f"In total {len(dataset)} samples")
    print(f"Evaluating with batch size {batch_size}")

    # --- Loop through prompts and run generation ---
    for batch in tqdm(dataset.iter(batch_size=1), desc="Generating and Collecting Data"):
        prompt = batch["prompt"][0]
        filename = batch["filename"][0]

        # Clear the model's internal cache for each new prompt
        model.caches_dirpath = caches_dirpath  # Set the directory for saving caches
        model.filename = filename  # Set the filename for saving caches

        with torch.no_grad():
            # Pass the calibration-enabled model to your generation function
            generated_image = gen_one_img(
                model, vae, text_tokenizer, text_encoder, prompt,
                g_seed=hash_str_to_int(filename),
                cfg_insertion_layer=[args.cfg_insertion_layer],
                scale_schedule=scale_schedule,
                cfg_list=[3.0] * len(scale_schedule),
                tau_list=[0.5] * len(scale_schedule),
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=0
            )
        
        if generated_image is not None and isinstance(generated_image, torch.Tensor):
             cv2.imwrite(os.path.join(samples_dirpath, f"{filename}.png"), generated_image.cpu().numpy())

    print(f"\n✅ Stateful calibration data saved to {caches_dirpath}")



@configclass
@dataclass
class CollectConfig:
    """Configuration for collecting calibration dataset.

    Args:
        root (`str`, *optional*, defaults to `"datasets"`):
            Root directory to save the collected dataset.
        dataset_name (`str`, *optional*, defaults to `"qdiff"`):
            Name of the collected dataset.
        prompt_path (`str`, *optional*, defaults to `"prompts/qdiff.yaml"`):
            Path to the prompt file.
        num_samples (`int`, *optional*, defaults to `128`):
            Number of samples to collect.
    """

    root: str = "datasets"
    dataset_name: str = "qdiff"
    data_path: str = "prompts/qdiff.yaml"
    num_samples: int = 128


if __name__ == "__main__":
    parser = DiffusionPtqRunConfig.get_parser()
    parser.add_config(CollectConfig, scope="collect", prefix="collect")
    configs, _, unused_cfgs, unused_args, unknown_args = parser.parse_known_args()
    ptq_config, collect_config = configs[""], configs["collect"]
    assert isinstance(ptq_config, DiffusionPtqRunConfig)
    assert isinstance(collect_config, CollectConfig)
    if len(unused_cfgs) > 0:
        print(f"Warning: unused configurations {unused_cfgs}")
    if unused_args is not None:
        print(f"Warning: unused arguments {unused_args}")
    assert len(unknown_args) == 0, f"Unknown arguments: {unknown_args}"

    collect_dirpath = os.path.join(
        collect_config.root,
        str(ptq_config.pipeline.dtype),
        ptq_config.pipeline.name,
        ptq_config.eval.protocol,
        collect_config.dataset_name,
        f"s{collect_config.num_samples}",
    )
    print(f"Saving caches to {collect_dirpath}")

    dataset = get_dataset(
        collect_config.data_path,
        max_dataset_size=collect_config.num_samples,
        return_gt=ptq_config.pipeline.task in ["canny-to-image"],
        repeat=1,
    )

    ptq_config.output.root = collect_dirpath
    os.makedirs(ptq_config.output.root, exist_ok=True)
    create_stateful_cache(ptq_config, dataset=dataset)
