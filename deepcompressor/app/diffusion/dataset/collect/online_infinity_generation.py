import os
import time
from dataclasses import dataclass
import cv2

import sys
sys.path.append('./Infinity_rep/')

import datasets
import torch
from omniconfig import configclass
from torch import nn
from tqdm import tqdm

from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.utils.common import hash_str_to_int, tree_map, tree_split
from infinity.models.infinity import Infinity 
from Infinity_rep.tools.run_infinity import *

from deepcompressor.app.diffusion.dataset.data import get_dataset

from infinity.models.infinity import Infinity, CrossAttnBlock, sample_with_top_k_top_p_also_inplace_modifying_logits_
from tools.run_infinity import load_visual_tokenizer, load_tokenizer, gen_one_img
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from contextlib import contextmanager
from deepcompressor.app.diffusion.dataset.collect.calib import CollectConfig

import inspect
import typing as tp
import functools

import argparse
import gc

model_path = '/workspace/Infinity/weights/infinity_2b_reg.pth'
vae_path = '/workspace/Infinity/weights/infinity_vae_d32reg.pth'
text_encoder_ckpt = '/workspace/Infinity/weights/flan-t5-xl'
h_div_w = 1/1 
enable_positive_prompt = 0

args_2b = argparse.Namespace(
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
    bf16=0,
    save_file='tmp.jpg',
    enable_model_cache=0
)
args_8b=argparse.Namespace(
    pn='1M', model_path='/workspace/deepcompressor/Infinity_rep/weights/infinity_8b_weights',
    vae_path='/workspace/Infinity/weights/infinity_vae_d56_f8_14_patchify.pth',
    text_encoder_ckpt='/workspace/Infinity/weights/flan-t5-xl',
    cfg_insertion_layer=0, vae_type=14, add_lvl_embeding_only_first_block=1,
    use_bit_label=1, model_type='infinity_8b', rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2, use_scale_schedule_embedding=0, sampling_per_bits=1,
    text_channels=2048, apply_spatial_patchify=1, h_div_w_template=1.000,
    use_flex_attn=0, cache_dir='/dev/shm', checkpoint_type='torch_shard',
    bf16=1, save_file='tmp.jpg'
)

h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_]['1M']['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

class StatefulInfinity(Infinity):
    """
    An Infinity model subclass designed for stateful data collection.
    It hooks into the autoregressive loop to capture the complete
    state required to replay a forward pass at any given step.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This will store the state for ONE image generation.
        self.collected_cache = []
        self.block_index = 0
        self.module_index = 0
        self.capture_schedule = None

    def set_block(self, block_index, module_index):
        self.block_index = block_index
        self.module_index = module_index

    def set_capture_schedule(self, schedule: dict[int, int], capture_once=False):
        """
        Define how many times to capture each VAR scale step.

        Args:
            schedule: dict[int, int]
                Mapping from scale index (si) -> number of remaining captures.
                Example: {0:64, 1:64, 2:64, 3:64, 4:12, 5:12, ...}
            capture_once: bool
                If True, stop capturing after the schedule is exhausted.
        """
        self.capture_schedule = dict(schedule)
        self.capture_once = bool(capture_once)
        self._capture_done = False

    def _should_capture_step(self, si: int) -> bool:
        """
        Return True if this scale step should be captured now.
        """
        if not hasattr(self, "capture_schedule") or self.capture_schedule is None:
            # fallback: capture all steps
            return not getattr(self, "_capture_done", False)
        return self.capture_schedule.get(si, 0) > 0


    def _update_capture_counter(self, si: int):
        """
        Decrement the counter for step `si` after one capture.
        """
        if not hasattr(self, "capture_schedule") or self.capture_schedule is None:
            return
        if si in self.capture_schedule:
            self.capture_schedule[si] -= 1
            if self.capture_schedule[si] <= 0:
                del self.capture_schedule[si]
        # stop if all done
        if self.capture_once and not self.capture_schedule:
            self._capture_done = True

    def reset_cache(self):
        """
        Clears the collected input data and attempts to release unused GPU memory
        to prevent memory leaks between multiple runs.
        """
        self.collected_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
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
        for si, pn in enumerate(scale_schedule):   # si: i-th segment
            cfg = cfg_list[si]
            capturing_this_step = self._should_capture_step(si)
            if si >= trunk_scale:
                break
            cur_L += np.array(pn).prod()

            need_to_pad = 0
            attn_fn = None
            if self.use_flex_attn:
                attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)

            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            layer_idx = 0
            for block_idx, b in enumerate(self.block_chunks):
                # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
                if self.add_lvl_embeding_only_first_block and block_idx == 0:
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                if not self.add_lvl_embeding_only_first_block: 
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                
                for m_idx, m in enumerate(b.module):
                    if m_idx == self.module_index and block_idx == self.block_index:
                        # Dictionary to store the inputs for the current module pass
                        calibration_data = {}
                        # List to hold hook handles for later removal
                        handles = []

                        # --- Capture the contextual kwargs needed for evaluation ---
                        # The cross-attention context is available in this scope
                        eval_kwargs_for_this_step = {'ca_kv': ca_kv}
                        
                        # The self-attention KV cache is stored inside the module.
                        # We capture its state BEFORE it's updated with the current step's data.
                        # The PatchedSelfAttention forward pass is designed to accept this structure.
                        sa_cache_k = m.sa.cached_k
                        sa_cache_v = m.sa.cached_v
                        
                        # We use clone().detach() to prevent any modifications to the saved cache
                        eval_kwargs_for_this_step['sa_kv_cache'] = {
                            'sa': {
                                'k': sa_cache_k.clone().detach() if sa_cache_k is not None else None,
                                'v': sa_cache_v.clone().detach() if sa_cache_v is not None else None
                            }
                        }

                        # Also capture any other non-tensor args needed by the forward pass
                        eval_kwargs_for_this_step['scale_schedule'] = scale_schedule
                        eval_kwargs_for_this_step['scale_ind'] = si

                        # Store this kwargs dictionary in our main data package
                        calibration_data['eval_kwargs'] = eval_kwargs_for_this_step
                        
                        # Define the generic hook function to capture the input tensor
                        def get_input_hook(name):
                            def hook(model, input, output):
                                # Input to a nn.Linear module is a tuple. We capture the first element.
                                calibration_data[name] = input[0].detach()
                            return hook
                        
                        # Attach hooks to the PatchedSelfAttention linear layers
                        handles.append(m.sa.to_q.register_forward_hook(get_input_hook('sa_q')))
                        handles.append(m.sa.to_k.register_forward_hook(get_input_hook('sa_k')))
                        handles.append(m.sa.to_v.register_forward_hook(get_input_hook('sa_v')))
                        handles.append(m.sa.to_out[0].register_forward_hook(get_input_hook('sa_out')))

                        # Attach hooks to the PatchedCrossAttention linear layers
                        handles.append(m.ca.to_q.register_forward_hook(get_input_hook('ca_q')))
                        handles.append(m.ca.to_k.register_forward_hook(get_input_hook('ca_k')))
                        handles.append(m.ca.to_v.register_forward_hook(get_input_hook('ca_v')))
                        handles.append(m.ca.to_out[0].register_forward_hook(get_input_hook('ca_out')))
                        
                        # Attach hooks to the FFN linear layers
                        if hasattr(m.ffn, 'fcg'):  # For FFNSwiGLU
                            handles.append(m.ffn.fcg.register_forward_hook(get_input_hook('ffn_fcg')))
                        handles.append(m.ffn.fc1.register_forward_hook(get_input_hook('ffn_fc1')))
                        handles.append(m.ffn.fc2.register_forward_hook(get_input_hook('ffn_fc2')))

                        # Add hooks to the container modules to capture their direct inputs for evaluation
                        handles.append(m.sa.register_forward_hook(get_input_hook('sa')))
                        handles.append(m.ca.register_forward_hook(get_input_hook('ca')))


                    last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, scale_ind=si)
                    if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                        # print(f'add cfg={cfg} on {layer_idx}-th layer output')
                        last_stage = cfg * last_stage[:B] + (1-cfg) * last_stage[B:]
                        last_stage = torch.cat((last_stage, last_stage), 0)
                    layer_idx += 1
                    
                    if m_idx == self.module_index and block_idx == self.block_index:
                        # The forward pass of module 'm' has just run, triggering the hooks.
                        # Now, save the collected data and clean up.
                        #self.collected_cache.append(calibration_data)
                        if capturing_this_step and (m_idx == self.module_index) and (block_idx == self.block_index):
                            self.collected_cache.append(calibration_data)
                            for handle in handles:
                                handle.remove()
                            # decrement the step counter if scheduled
                            if self.capture_schedule and si in self.capture_schedule:
                                self.capture_schedule[si] -= 1
                                if self.capture_schedule[si] <= 0:
                                    del self.capture_schedule[si]
                            if self.capture_once and not self.capture_schedule:
                                self._capture_done = True

                        # Remove all the hooks to prevent them from firing again
                        for handle in handles:
                            handle.remove()
            
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
                h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl).float()   # BlC

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
    with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
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


def get_stateful_cache(model: Infinity, config: DiffusionPtqRunConfig, pipeline_config: dict, 
                       dataset: datasets.Dataset, block_idx: int, module_idx: int, save_kv_cache_only: bool = False, save_imgs: bool = False):
    model.set_block(block_idx, module_idx)
    if config.pipeline.name == 'infinity_2b':
        args = args_2b
    elif config.pipeline.name == 'infinity_8b':
        args = args_8b
    else:
        raise NotImplementedError(f"Pipeline {config.pipeline.name} not implemented")
    vae = load_visual_tokenizer(args)
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    print(f"In total {len(dataset)} samples")

    all_final_entries = [] 

    # --- Loop through prompts and run generation ---
    #schedule = {si: (64 if si < 10 else 32) for si in range(len(scale_schedule))}
    schedule = {si: (32 if si < 10 else 16) for si in range(len(scale_schedule))}

    # Apply it to the model
    model.set_capture_schedule(schedule)
    
    for batch in tqdm(dataset.iter(batch_size=1), desc="Generating and Collecting Data"):
        prompt = batch["prompt"][0]
        filename = batch["filename"][0]

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
        if save_kv_cache_only:
            # keep ONLY the final KV cache for this generation
            final_entry = None
            for d in reversed(model.collected_cache):
                ek = d.get("eval_kwargs", {})
                kv = ek.get("sa_kv_cache", {})
                sa = kv.get("sa", {}) if isinstance(kv, dict) else {}
                if "k" in sa or "v" in sa:
                    final_entry = {
                        "sa_k_final": sa.get("k").detach() if torch.is_tensor(sa.get("k")) else None,
                        "sa_v_final": sa.get("v").detach() if torch.is_tensor(sa.get("v")) else None,
                    }
                    break

            if final_entry is not None:
                all_final_entries.append(final_entry)

            # clear per-step scratch from the model for the next image
            model.collected_cache.clear()
            torch.cuda.empty_cache(); gc.collect()


        if save_imgs:
            os.makedirs('./temp_imgs', exist_ok=True)
            cv2.imwrite(f'./temp_imgs/temp_{str(time.time())}.png', generated_image.detach().cpu().numpy())

    result = model.collected_cache
    model.collected_cache = []
    model.reset_cache()
    if save_kv_cache_only:
        result = all_final_entries
    return result


if __name__ == "__main__":
    from deepcompressor.app.diffusion.nn.struct_infinity import patchModel 

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

    vae = load_visual_tokenizer(args)
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    stateful_model = load_transformer(vae, args)
    model = patchModel(stateful_model) # Use the patching function from your struct file

    ptq_config.output.root = collect_dirpath
    os.makedirs(ptq_config.output.root, exist_ok=True)
    cache = get_stateful_cache(model, ptq_config, dataset=dataset, block_idx=2, module_idx=1)