import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
from dataclasses import dataclass, field
from functools import partial
import math
import sys
import argparse 
import cv2
import numpy as np

from flash_attn import flash_attn_varlen_kvpacked_func

# --- Configure Python Path to Find Modules ---
sys.path.append('./') 
sys.path.append('./Infinity_rep') 

# --- Real Imports from DeepCompressor and Infinity Libraries ---

# DeepCompressor Structs
from deepcompressor.nn.struct.base import BaseModuleStruct
from deepcompressor.nn.struct.attn import AttentionConfigStruct, AttentionStruct
from deepcompressor.utils.common import join_name
from deepcompressor.app.diffusion.nn.struct import (
    DiffusionAttentionStruct,
    DiffusionFeedForwardStruct,
    DiffusionTransformerBlockStruct,
    FeedForwardConfigStruct,
    DiffusionModuleStruct,
    DiTStruct,
)

# Infinity Model Components
from diffusers.models.attention_processor import Attention
from infinity.models.infinity import Infinity
from infinity.models.basic import (
    CrossAttnBlock,
    SelfAttention,
    CrossAttention,
    FFN,
    FFNSwiGLU,
    apply_rotary_emb
)
# We need to import the loader functions
from tools.run_infinity import load_visual_tokenizer, load_transformer, load_tokenizer, gen_one_img, h_div_w_templates, dynamic_resolution_h_w


# --- MODIFICATION: We will create a modified SelfAttention layer ---
# This new layer will be compatible with the deepcompressor framework.

class PatchedSelfAttention(Attention):
    """
    Inherits from diffusers.Attention to be fully compatible with deepcompressor.
    It overrides the __init__ and forward methods to replicate the exact behavior
    of the original Infinity SelfAttention layer with unfused projections.
    """
    def __init__(self, original_sa_module: SelfAttention, module_name=None):
        dim_head = original_sa_module.proj.in_features // original_sa_module.num_heads
        super().__init__(
            query_dim=original_sa_module.proj.in_features,
            heads=original_sa_module.num_heads,
            dim_head=dim_head,
            bias=True
        )
        
        self.module_name = module_name
        original_qkv_weight = original_sa_module.mat_qkv.weight
        q_w, k_w, v_w = torch.chunk(original_qkv_weight, 3, dim=0)
        self.to_q.weight.data.copy_(q_w)
        self.to_k.weight.data.copy_(k_w)
        self.to_v.weight.data.copy_(v_w)
        
        self.to_q.bias.data.copy_(original_sa_module.q_bias)
        self.to_k.bias.data.zero_()
        self.to_v.bias.data.copy_(original_sa_module.v_bias)
        
        self.to_out[0].weight.data.copy_(original_sa_module.proj.weight)
        self.to_out[0].bias.data.copy_(original_sa_module.proj.bias)

        self.cos_attn = original_sa_module.cos_attn
        if self.cos_attn:
            self.scale = 1.0
            self.scale_mul_1H11 = nn.Parameter(original_sa_module.scale_mul_1H11.data.clone())
            self.max_scale_mul = original_sa_module.max_scale_mul
        else:
            tau = getattr(original_sa_module, 'tau', 1.0)
            self.scale = 1 / math.sqrt(dim_head) / tau
        
        self.pad_to_multiplier = original_sa_module.pad_to_multiplier
        self.rope2d_normalized_by_hw = original_sa_module.rope2d_normalized_by_hw
        
        self.caching = False
        self.cached_k = None
        self.cached_v = None

        self.last_k = None
        self.last_v = None

        self.proj = self.to_out[0]

    def kv_caching(self, enable: bool):
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

    def forward(self, hidden_states, attention_mask=None, attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0, **kwargs):
        """
        A robust forward method that handles potentially missing non-tensor arguments.
        """
        sa_kv_cache = kwargs.get('sa_kv_cache', {})

        #  Get the specific cache for THIS module using its key 'sa'
        past_k, past_v = None, None
        if sa_kv_cache:
            cache_for_this_module = sa_kv_cache.get('sa', {})
            past_k = cache_for_this_module.get('k')
            past_v = cache_for_this_module.get('v')

        B, L_current, C = hidden_states.shape
        head_dim = self.inner_dim // self.heads
        
        q = self.to_q(hidden_states).view(B, L_current, self.heads, head_dim).transpose(1, 2)
        k = self.to_k(hidden_states).view(B, L_current, self.heads, head_dim).transpose(1, 2)
        v = self.to_v(hidden_states).view(B, L_current, self.heads, head_dim).transpose(1, 2)
        
        if self.cos_attn:
            scale_mul = self.scale_mul_1H11.reshape(1, self.heads, 1, 1).clamp_max(self.max_scale_mul).exp()
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul)
            k = F.normalize(k, dim=-1, eps=1e-12)

        if rope2d_freqs_grid is not None and scale_schedule is not None:
            q, k = apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind)

        # Cache can be stored internally or received through kwargs
        # To keep compatibility with both Infinity and Deepcompressor
        if self.caching:
            if self.cached_k is None:
                self.cached_k, self.cached_v = k, v
            else:
                self.cached_k = torch.cat((self.cached_k, k), dim=2)
                self.cached_v = torch.cat((self.cached_v, v), dim=2)
            k_final, v_final = self.cached_k, self.cached_v
        else:
            if past_k is not None:
                k_final = torch.cat([past_k, k], dim=2)
                v_final = torch.cat([past_v, v], dim=2)
            else:
                k_final, v_final = k, v
            
        out = F.scaled_dot_product_attention(q, k_final, v_final, attn_mask=attention_mask, scale=self.scale)
        
        out = out.transpose(1, 2).reshape(B, L_current, C)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        
        return out

class PatchedCrossAttention(Attention):
    """
    Inherits from diffusers.Attention but now uses the correct variable-length
    attention mechanism from the original Infinity model.
    """
    def __init__(self, original_ca_module: CrossAttention):
        dim_head = original_ca_module.head_dim
        super().__init__(
            query_dim=original_ca_module.embed_dim,
            cross_attention_dim=original_ca_module.kv_dim,
            heads=original_ca_module.num_heads,
            dim_head=dim_head,
            bias=True
        )
        
        self.dim_head = dim_head

        self.to_q.weight.data.copy_(original_ca_module.mat_q.weight)
        self.to_q.bias.data.copy_(original_ca_module.mat_q.bias)
        
        original_kv_weight = original_ca_module.mat_kv.weight
        k_w, v_w = torch.chunk(original_kv_weight, 2, dim=0)
        self.to_k.weight.data.copy_(k_w)
        self.to_v.weight.data.copy_(v_w)
        
        self.to_k.bias = None
        self.to_v.bias.data.copy_(original_ca_module.v_bias)
        
        self.to_out[0].weight.data.copy_(original_ca_module.proj.weight)
        self.to_out[0].bias.data.copy_(original_ca_module.proj.bias)

        self.scale = 1 / math.sqrt(dim_head)
        self.proj = self.to_out[0]

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        """
        This forward pass now correctly handles the variable-length context (`ca_kv`)
        by using the specialized flash_attn_varlen_kvpacked_func, mimicking the
        original model's behavior and avoiding the shape error.
        """
        ca_kv = kwargs.get('ca_kv')
        if encoder_hidden_states is None and ca_kv is not None:
            encoder_hidden_states = ca_kv
            
        if not isinstance(encoder_hidden_states, tuple):
            raise ValueError("PatchedCrossAttention expects encoder_hidden_states to be the ca_kv tuple.")

        ca_kv = encoder_hidden_states
        kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        kv_compact = kv_compact.float()
        
        B_x, L_x, C = hidden_states.shape
        
        q_compact = self.to_q(hidden_states).view(-1, self.heads, self.dim_head)

        k_compact = self.to_k(kv_compact).view(-1, self.heads, self.dim_head)
        v_compact = self.to_v(kv_compact).view(-1, self.heads, self.dim_head)
        
        kv_packed = torch.stack([k_compact, v_compact], dim=1).contiguous()
        
        cu_seqlens_q = torch.arange(0, L_x * (B_x + 1), L_x, dtype=torch.int32, device=hidden_states.device)

        # --- Cast inputs to bfloat16 for FlashAttention compatibility ---
        oup = flash_attn_varlen_kvpacked_func(
            q=q_compact.to(torch.bfloat16), 
            kv=kv_packed.to(torch.bfloat16), 
            cu_seqlens_q=cu_seqlens_q.to(torch.int32), 
            cu_seqlens_k=cu_seqlens_k.to(torch.int32), 
            max_seqlen_q=L_x, 
            max_seqlen_k=max_seqlen_k, 
            dropout_p=0, 
            softmax_scale=self.scale
        ).reshape(B_x, L_x, -1).float()
        
        # Output projection
        out = self.to_out[0](oup)
        out = self.to_out[1](out)
        
        # Return the output in the same dtype as the input for consistency
        return out.to(hidden_states.dtype)
    
class InfinityAttentionStruct(DiffusionAttentionStruct):
    @staticmethod
    def _default_construct(
        module: Attention, # Now expects a diffusers.Attention subclass
        /, parent: tp.Optional["InfinityTransformerBlockStruct"] = None,
        fname: str = "", rname: str = "", rkey: str = "", idx: int = 0, **kwargs,
    ) -> "InfinityAttentionStruct":
        
        is_cross = module.cross_attention_dim is not None

        if hasattr(module, 'proj'):
             o_proj, o_proj_rname = module.proj, "proj"
        else:
             o_proj, o_proj_rname = module.to_out[0], "to_out.0"

        # CORRECTED: Use `module.heads` which is the attribute from the parent Attention class
        num_heads = module.heads
        
        config = AttentionConfigStruct(
            hidden_size=o_proj.in_features,
            inner_size=o_proj.in_features,
            num_query_heads=num_heads, 
            num_key_value_heads=num_heads,
            with_qk_norm=getattr(module, 'cos_attn', False),
            with_rope=not is_cross,
            linear_attn=False,
        )
        
        return InfinityAttentionStruct(
            module=module, parent=parent, fname=fname, idx=idx, rname=rname, rkey=rkey,
            config=config, q_proj=module.to_q, k_proj=module.to_k, v_proj=module.to_v, o_proj=o_proj,
            q_proj_rname="to_q", k_proj_rname="to_k", v_proj_rname="to_v",
            o_proj_rname=o_proj_rname, add_q_proj=None, add_k_proj=None, add_v_proj=None,
            add_o_proj=None, add_q_proj_rname="", add_k_proj_rname="", add_v_proj_rname="",
            add_o_proj_rname="", q=None, k=None, v=None, q_rname="", k_rname="", v_rname=""
        )

    def filter_kwargs(self, kwargs: dict) -> dict:
        """
        Overrides the base class method. Returns the kwargs dictionary
        without filtering to ensure custom arguments like 'ca_kv' are
        passed through to the evaluation module.
        """
        return kwargs


class InfinityFeedForwardStruct(DiffusionFeedForwardStruct):
    @staticmethod
    def _default_construct(module: tp.Union[FFN, FFNSwiGLU],/, parent: tp.Optional["InfinityTransformerBlockStruct"] = None,fname: str = "", rname: str = "", rkey: str = "", idx: int = 0, **kwargs,) -> "InfinityFeedForwardStruct":
        if isinstance(module, FFN):
            up_projs,down_projs,up_proj_rnames,down_proj_rnames,act_type = [module.fc1],[module.fc2],["fc1"],["fc2"],"gelu"
        elif isinstance(module, FFNSwiGLU):
            up_projs,down_projs,up_proj_rnames,down_proj_rnames,act_type = [module.fcg, module.fc1],[module.fc2],["fcg", "fc1"],["fc2"],"swiglu"
        else:
            raise TypeError(f"Unsupported module type: {type(module)}")
        config = FeedForwardConfigStruct(hidden_size=down_projs[0].out_features, intermediate_size=up_projs[0].out_features, intermediate_act_type=act_type, num_experts=1)
        
        return DiffusionFeedForwardStruct(
            module=module, parent=parent, fname=fname, idx=idx, rname=rname, rkey=rkey,
            config=config, up_projs=up_projs, down_projs=down_projs,
            up_proj_rnames=up_proj_rnames, down_proj_rnames=down_proj_rnames
        )
   
class InfinityTransformerBlockStruct(DiffusionTransformerBlockStruct):
    def __post_init__(self):
        self.attn_struct_cls = InfinityAttentionStruct
        self.ffn_struct_cls = InfinityFeedForwardStruct
        super().__post_init__()

    @staticmethod
    def _default_construct(module: CrossAttnBlock,/, parent: tp.Optional[BaseModuleStruct] = None,fname: str = "", rname: str = "", rkey: str = "", idx: int = 0, **kwargs,) -> "InfinityTransformerBlockStruct":
        return InfinityTransformerBlockStruct(
            module=module, parent=parent, fname=fname, idx=idx, rname=rname, rkey=rkey,
            attns=[module.sa, module.ca],
            ffn=module.ffn,
            pre_attn_norms=[module.ln_wo_grad, module.ca_norm],
            pre_ffn_norm=module.ln_wo_grad,
            parallel=False, 
            pre_attn_add_norms=[],
            add_ffn=None,
            pre_add_ffn_norm=None,
            norm_type="layer_norm", 
            add_norm_type="layer_norm",
            pre_attn_norm_rnames=["ln_wo_grad", "ca_norm"],
            attn_rnames=["sa", "ca"],
            pre_ffn_norm_rname="ln_wo_grad",
            ffn_rname="ffn",
            pre_attn_add_norm_rnames=[],
            pre_add_ffn_norm_rname="",
            add_ffn_rname=""
        )

class InfinityStruct(DiTStruct):
    """ Top-level adapter for the entire Infinity model with corrected naming logic. """
    transformer_block_struct_cls: tp.ClassVar[type[DiffusionTransformerBlockStruct]] = InfinityTransformerBlockStruct
    
    def __post_init__(self):
        super(DiTStruct, self).__post_init__()

        # --- FIX: Manually set the _name attributes on self ---
        self.pre_module_structs = {}
        for fname in ("input_embed", "time_embed", "text_embed"):
            module, rname, rkey = getattr(self, fname), getattr(self, f"{fname}_rname"), getattr(self, f"{fname}_rkey")
            setattr(self, f"{fname}_key", join_name(self.key, rkey, sep="_"))
            if module is not None:
                name = join_name(self.name, rname)
                # This is the new, critical line that was missing
                setattr(self, f"{fname}_name", name) 
                self.pre_module_structs.setdefault(name, DiffusionModuleStruct(module=module, parent=self, fname=fname, rname=rname, rkey=rkey))

        self.post_module_structs = {}
        self.norm_out_key = join_name(self.key, self.norm_out_rkey, sep="_")
        for fname in ("norm_out", "proj_out"):
             module, rname, rkey = getattr(self, fname), getattr(self, f"{fname}_rname"), getattr(self, f"{fname}_rkey")
             if module is not None:
                 name = join_name(self.name, rname)
                 # This is the new, critical line that was missing
                 setattr(self, f"{fname}_name", name)
                 self.post_module_structs.setdefault(name, DiffusionModuleStruct(module=module, parent=self, fname=fname, rname=rname, rkey=rkey))

        # --- CUSTOM BLOCK NAMING LOGIC (from previous fix) ---
        num_chunks = self.module.num_block_chunks
        num_blocks_per_chunk = self.module.num_blocks_in_a_chunk
        transformer_block_rnames = [f"{self.transformer_blocks_rname}.{i}.module.{j}" for i in range(num_chunks) for j in range(num_blocks_per_chunk)]
        
        self.transformer_blocks_name = join_name(self.name, self.transformer_blocks_rname) # "block_chunks"
        self.transformer_block_names = [join_name(self.name, rname) for rname in transformer_block_rnames]
        
        self.transformer_block_structs = [
            self.transformer_block_struct_cls.construct(
                layer, parent=self, fname="transformer_block", rname=rname, rkey=self.transformer_block_rkey, idx=idx,
            )
            for idx, (layer, rname) in enumerate(zip(self.transformer_blocks, transformer_block_rnames, strict=True))
        ]

    @staticmethod
    def _default_construct(
        module: Infinity,
        /, parent: tp.Optional[BaseModuleStruct] = None,
        fname: str = "", rname: str = "", rkey: str = "", idx: int = 0, **kwargs,
    ) -> "InfinityStruct":
        all_blocks = [m for chunk in module.block_chunks for m in chunk.module]
        transformer_blocks_list = nn.ModuleList(all_blocks)
        return InfinityStruct(
            module=module, parent=parent, fname=fname, idx=idx, rname=rname, rkey=rkey,
            input_embed=module.word_embed, time_embed=module.shared_ada_lin, text_embed=module.text_proj_for_ca,
            transformer_blocks=transformer_blocks_list, norm_out=module.head_nm, proj_out=module.head,
            input_embed_rname="word_embed", time_embed_rname="shared_ada_lin", text_embed_rname="text_proj_for_ca",
            transformer_blocks_rname="block_chunks", norm_out_rname="head_nm", proj_out_rname="head",
        )

DiffusionAttentionStruct.register_factory((SelfAttention, CrossAttention), InfinityAttentionStruct._default_construct)
DiffusionFeedForwardStruct.register_factory((FFN, FFNSwiGLU), InfinityFeedForwardStruct._default_construct)
DiffusionTransformerBlockStruct.register_factory(CrossAttnBlock, InfinityTransformerBlockStruct._default_construct)
DiTStruct.register_factory(Infinity, InfinityStruct._default_construct)
DiffusionAttentionStruct.register_factory((PatchedSelfAttention, PatchedCrossAttention), InfinityAttentionStruct._default_construct)
# And as a fallback, register with the absolute base class as you suggested.
BaseModuleStruct.register_factory(Infinity, InfinityStruct._default_construct)


def patchModel(model: Infinity) -> nn.Module:
    """
    Replaces the original attention layers with unfused, compatible versions,
    ensuring they are on the same device as the original model.
    """
    for block in model.block_chunks.children():
        for sub_block_name, sub_block in block.module.named_children():
            if isinstance(sub_block, CrossAttnBlock):    
                # 1. Get the correct device from the original module before replacing it.
                device = sub_block.sa.proj.weight.device
                
                # 2. Create the new patched modules and immediately move them to the correct device.
                patched_sa = PatchedSelfAttention(sub_block.sa).to(device)
                patched_ca = PatchedCrossAttention(sub_block.ca).to(device)
                
                # 3. Assign the new, device-correct modules back to the model.
                sub_block.sa = patched_sa
                sub_block.ca = patched_ca
                
    return model



# --- Main Test Execution ---
def main():
    print("--- Loading a real Infinity model from checkpoint ---")
    
    args = argparse.Namespace(
        pn='1M', model_path='/workspace/Infinity/weights/infinity_2b_reg.pth',
        vae_path='/workspace/Infinity/weights/infinity_vae_d32reg.pth',
        text_encoder_ckpt='/workspace/Infinity/weights/flan-t5-xl',
        model_type='infinity_2b', vae_type=32, text_channels=2048,
        add_lvl_embeding_only_first_block=1, use_bit_label=1,
        rope2d_each_sa_layer=1, rope2d_normalized_by_hw=2, apply_spatial_patchify=0,
        cfg_insertion_layer=0, use_scale_schedule_embedding=0, sampling_per_bits=1,
        h_div_w_template=1.000, use_flex_attn=0, cache_dir='/dev/shm',
        checkpoint_type='torch', seed=0, bf16=1, save_file='tmp.jpg',
        enable_model_cache=0
    )
    
    vae = load_visual_tokenizer(args)
    model = load_transformer(vae, args)

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)

    h_div_w = 1/1
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    img = gen_one_img(
        model,
        vae,
        text_tokenizer,
        text_encoder,
        'A photo of a cat',
        g_seed=16,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=3.0,
        tau_list=0.5,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=True,
    )
    cv2.imwrite('img.png', img.detach().cpu().numpy())
    
    print("Full Infinity model loaded successfully.\n")
    # --- Patch the model before creating the struct ---
    print("--- Patching attention layers to be compatible ---")
    patched_model = patchModel(model)
    print("Patching complete.\n")


    img = gen_one_img(
        patched_model,
        vae,
        text_tokenizer,
        text_encoder,
        'A photo of a cat',
        g_seed=16,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=3.0,
        tau_list=0.5,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=True,
    )
    cv2.imwrite('img_patched.png', img.detach().cpu().numpy())
    

    # --- Test the full model parsing ---
    print("--- Testing full Infinity model parsing ---")
    
    # The generic construct call should now work because we've patched it.
    model_struct = DiTStruct.construct(patched_model)

    print(f"Created model_struct for: {type(model_struct.module).__name__}")
    print(f"  - The struct is of type: {type(model_struct).__name__}")
    print(f"  - It found {model_struct.num_blocks} transformer blocks.")
    
    assert isinstance(model_struct, DiTStruct)
    assert len(model_struct.block_structs) > 0
    first_block_struct = model_struct.block_structs[0]
    
    print(f"  - First block is of type: {type(first_block_struct).__name__}")
    assert isinstance(first_block_struct, InfinityTransformerBlockStruct)
    
    assert first_block_struct.parent is model_struct
    assert first_block_struct.attn_structs[0].parent is first_block_struct

    print("\n✅ Verification for the full Infinity model struct successful!\n")

if __name__ == "__main__":
    main()