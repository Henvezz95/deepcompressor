import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
from dataclasses import dataclass, field
from functools import partial
import math
import sys
import argparse 

# --- Configure Python Path to Find Modules ---
sys.path.append('./') 
sys.path.append('./Infinity_rep') 

# --- Real Imports from DeepCompressor and Infinity Libraries ---

# DeepCompressor Structs
from deepcompressor.nn.struct.base import BaseModuleStruct
from deepcompressor.nn.struct.attn import AttentionConfigStruct
from deepcompressor.app.diffusion.nn.struct import (
    DiffusionAttentionStruct,
    DiffusionFeedForwardStruct,
    DiffusionTransformerBlockStruct,
    FeedForwardConfigStruct,
    DiTStruct,
)

# Infinity Model Components
from infinity.models.infinity import Infinity
from infinity.models.basic import (
    CrossAttnBlock,
    SelfAttention,
    CrossAttention,
    FFN,
    FFNSwiGLU
)
# We need to import the loader functions
from tools.run_infinity import load_visual_tokenizer, load_transformer


# --- Our New Adapter Structs (These are correct and remain the same) ---

class InfinityAttentionStruct(DiffusionAttentionStruct):
    def __post_init__(self):
        BaseModuleStruct.__post_init__(self)
        if isinstance(self.module, SelfAttention):
            assert self.q_proj.weight.shape[0] == self.config.hidden_size * 3
        else:
            assert self.q_proj.weight.shape[0] == self.config.hidden_size
            assert self.k_proj.weight.shape[0] == self.config.hidden_size * 2
        assert self.o_proj.weight.shape[1] == self.config.hidden_size

    @staticmethod
    def _default_construct(module: tp.Union[SelfAttention, CrossAttention],/, parent: tp.Optional["InfinityTransformerBlockStruct"] = None,fname: str = "", rname: str = "", rkey: str = "", idx: int = 0, **kwargs,) -> "InfinityAttentionStruct":
        if isinstance(module, SelfAttention):
            q_proj, k_proj, v_proj = module.mat_qkv, module.mat_qkv, module.mat_qkv
            q_proj_rname, k_proj_rname, v_proj_rname = "mat_qkv", "mat_qkv", "mat_qkv"
            o_proj, o_proj_rname = module.proj, "proj"
            with_rope = True
        elif isinstance(module, CrossAttention):
            q_proj, q_proj_rname = module.mat_q, "mat_q"
            k_proj, v_proj = module.mat_kv, module.mat_kv
            k_proj_rname, v_proj_rname = "mat_kv", "mat_kv"
            o_proj, o_proj_rname = module.proj, "proj"
            with_rope = False
        else:
            raise TypeError(f"Unsupported module type: {type(module)}")
        config = AttentionConfigStruct(hidden_size=o_proj.in_features,inner_size=q_proj.weight.shape[0],num_query_heads=module.num_heads, num_key_value_heads=module.num_heads,with_qk_norm=False, with_rope=with_rope, linear_attn=False)
        return InfinityAttentionStruct(module=module, parent=parent, fname=fname, idx=idx, rname=rname, rkey=rkey,config=config, q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, o_proj=o_proj,q_proj_rname=q_proj_rname, k_proj_rname=k_proj_rname, v_proj_rname=v_proj_rname,o_proj_rname=o_proj_rname, add_q_proj=None, add_k_proj=None, add_v_proj=None,add_o_proj=None, add_q_proj_rname="", add_k_proj_rname="", add_v_proj_rname="",add_o_proj_rname="", q=None, k=None, v=None, q_rname="", k_rname="", v_rname="")

class InfinityFeedForwardStruct(DiffusionFeedForwardStruct):
    def __post_init__(self):
        BaseModuleStruct.__post_init__(self)
        for up_proj in self.up_projs:
            assert up_proj.weight.shape[1] == self.config.hidden_size
        for down_proj in self.down_projs:
            assert down_proj.weight.shape[0] == self.config.hidden_size

    @staticmethod
    def _default_construct(module: tp.Union[FFN, FFNSwiGLU],/, parent: tp.Optional["InfinityTransformerBlockStruct"] = None,fname: str = "", rname: str = "", rkey: str = "", idx: int = 0, **kwargs,) -> "InfinityFeedForwardStruct":
        if isinstance(module, FFN):
            up_projs,down_projs,up_proj_rnames,down_proj_rnames,act_type = [module.fc1],[module.fc2],["fc1"],["fc2"],"gelu"
        elif isinstance(module, FFNSwiGLU):
            up_projs,down_projs,up_proj_rnames,down_proj_rnames,act_type = [module.fcg, module.fc1],[module.fc2],["fcg", "fc1"],["fc2"],"swiglu"
        else:
            raise TypeError(f"Unsupported module type: {type(module)}")
        config = FeedForwardConfigStruct(hidden_size=down_projs[0].out_features, intermediate_size=up_projs[0].out_features, intermediate_act_type=act_type, num_experts=1)
        return InfinityFeedForwardStruct(module=module, parent=parent, fname=fname, idx=idx, rname=rname, rkey=rkey,config=config, up_projs=up_projs, down_projs=down_projs,up_proj_rnames=up_proj_rnames, down_proj_rnames=down_proj_rnames)

class InfinityTransformerBlockStruct(DiffusionTransformerBlockStruct):
    def __post_init__(self):
        # We need to explicitly set the child struct classes for the parent
        self.attn_struct_cls = InfinityAttentionStruct
        self.ffn_struct_cls = InfinityFeedForwardStruct
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    @staticmethod
    def _default_construct(module: CrossAttnBlock,/, parent: tp.Optional[BaseModuleStruct] = None,fname: str = "", rname: str = "", rkey: str = "", idx: int = 0, **kwargs,) -> "InfinityTransformerBlockStruct":
        return InfinityTransformerBlockStruct(module=module, parent=parent, fname=fname, idx=idx, rname=rname, rkey=rkey,attns=[module.sa, module.ca], ffn=module.ffn,pre_attn_norms=[module.ln_wo_grad, module.ca_norm], pre_ffn_norm=module.ln_wo_grad,parallel=False, pre_attn_add_norms=[], add_ffn=None, pre_add_ffn_norm=None,norm_type="layer_norm", add_norm_type="layer_norm",pre_attn_norm_rnames=["ln_wo_grad", "ca_norm"], attn_rnames=["sa", "ca"],pre_ffn_norm_rname="ln_wo_grad", ffn_rname="ffn",pre_attn_add_norm_rnames=[], pre_add_ffn_norm_rname="", add_ffn_rname="")

class InfinityStruct(DiTStruct):
    """ Top-level adapter for the entire Infinity model. """
    transformer_block_struct_cls: tp.ClassVar[type[DiffusionTransformerBlockStruct]] = InfinityTransformerBlockStruct
    
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
            input_embed=module.word_embed,
            time_embed=module.shared_ada_lin,
            text_embed=module.text_proj_for_ca,
            transformer_blocks=transformer_blocks_list,
            norm_out=module.head_nm,
            proj_out=module.head,
            input_embed_rname="word_embed",
            time_embed_rname="shared_ada_lin",
            text_embed_rname="text_proj_for_ca",
            transformer_blocks_rname="block_chunks",
            norm_out_rname="head_nm",
            proj_out_rname="head",
        )

# --- The New Strategy: Comprehensive Registration ---

# Register all our custom adapters with their appropriate base classes.
# This ensures they can be found by the generic `construct` method.
DiffusionAttentionStruct.register_factory((SelfAttention, CrossAttention), InfinityAttentionStruct._default_construct)
DiffusionFeedForwardStruct.register_factory((FFN, FFNSwiGLU), InfinityFeedForwardStruct._default_construct)
DiffusionTransformerBlockStruct.register_factory(CrossAttnBlock, InfinityTransformerBlockStruct._default_construct)
# Registering with DiTStruct, which is the direct parent we want to use.
DiTStruct.register_factory(Infinity, InfinityStruct._default_construct)
# And as a fallback, register with the absolute base class as you suggested.
BaseModuleStruct.register_factory(Infinity, InfinityStruct._default_construct)


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
    
    print("Full Infinity model loaded successfully.\n")

    # --- Test the full model parsing ---
    print("--- Testing full Infinity model parsing ---")
    
    # The generic construct call should now work because we've registered our factories.
    model_struct = DiTStruct.construct(model)

    print(f"Created model_struct for: {type(model_struct.module).__name__}")
    print(f"  - The struct is of type: {type(model_struct).__name__}")
    print(f"  - It found {model_struct.num_blocks} transformer blocks.")
    
    # Verification checks
    assert isinstance(model_struct, InfinityStruct)
    assert len(model_struct.block_structs) > 0
    first_block_struct = model_struct.block_structs[0]
    
    print(f"  - First block is of type: {type(first_block_struct).__name__}")
    assert isinstance(first_block_struct, InfinityTransformerBlockStruct)
    
    assert first_block_struct.parent is model_struct
    assert first_block_struct.attn_structs[0].parent is first_block_struct

    print("\n✅ Verification for the full Infinity model struct successful!\n")

if __name__ == "__main__":
    main()
