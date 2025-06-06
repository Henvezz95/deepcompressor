# run_infinity_dev.py

import torch
import torch.nn as nn
import sys
import argparse
import typing as tp

# Add parent directory to path to import our modules
sys.path.append('.') 

# --- DeepCompressor Imports ---
# Import the base classes we will inherit from.
# The exact path might be slightly different, adjust if necessary.
from deepcompressor.nn.struct.base import BaseModuleStruct
from deepcompressor.nn.struct.attn import (
    AttentionConfigStruct,
    AttentionStruct,
    CrossAttentionStruct,
    FeedForwardConfigStruct,
    FeedForwardStruct,
    SelfAttentionStruct,
    TransformerBlockStruct,
    BaseTransformerStruct,
)
from deepcompressor.utils.common import join_name

# --- Your Project's Imports ---
try:
    from Infinity_rep.tools.run_infinity import load_visual_tokenizer, load_transformer
except ImportError:
    print("ERROR: Could not import 'load_visual_tokenizer' and 'load_transformer'.")
    print("Please ensure 'run_infinity.py' is in the project root or accessible in the Python path.")
    sys.exit(1)


# --- NEW Hierarchical Wrapper Classes using deepcompressor's Structs ---

class InfinityAttentionStruct(AttentionStruct):
    """A specific AttentionStruct for Infinity's attention blocks."""
    @classmethod
    def construct(cls, module: nn.Module, **kwargs) -> "InfinityAttentionStruct":
        # Here, we map the layers from an Infinity attention module (e.g., block.sa)
        # to the fields expected by AttentionStruct.
        # This requires knowing the attribute names inside your `CrossAttnBlock`'s `sa` and `ca`.
        # I am assuming standard names like `q_proj`, `k_proj`, etc. Please verify.
        
        is_cross_attn = hasattr(module, 'ca') # Heuristic to check if it's a cross-attention submodule
        
        # --- Create the configuration for this attention block ---
        # This should be populated with your model's actual hyperparameters
        attn_config = AttentionConfigStruct(
            hidden_size=module.dim,
            add_hidden_size=module.kv_dim if is_cross_attn else 0,
            inner_size=module.q_proj.out_features,
            num_query_heads=module.num_heads,
            num_key_value_heads=module.num_heads, # Assuming non-GQA for simplicity
        )

        if is_cross_attn:
            return CrossAttentionStruct(
                module=module, config=attn_config,
                q_proj=module.q_proj, o_proj=module.proj,
                add_k_proj=module.ca.k_proj, add_v_proj=module.ca.v_proj,
                q_proj_rname="q_proj", o_proj_rname="proj",
                add_k_proj_rname="ca.k_proj", add_v_proj_rname="ca.v_proj",
                # Other rnames can be empty strings if not present
                q_rname="", k_rname="", v_rname="",
                **kwargs
            )
        else: # Self-attention
             return SelfAttentionStruct(
                module=module, config=attn_config,
                q_proj=module.q_proj, k_proj=module.k_proj, 
                v_proj=module.v_proj, o_proj=module.proj,
                q_proj_rname="q_proj", k_proj_rname="k_proj",
                v_proj_rname="v_proj", o_proj_rname="proj",
                 # Other rnames can be empty strings if not present
                q_rname="", k_rname="", v_rname="",
                **kwargs
             )


class InfinityFFNStruct(FeedForwardStruct):
    """A specific FeedForwardStruct for Infinity's FFN blocks."""
    @classmethod
    def construct(cls, module: nn.Module, **kwargs) -> "InfinityFFNStruct":
        # Map layers from an Infinity FFN module (e.g., block.ffn) to FeedForwardStruct fields.
        # I am assuming names `fc1` and `fc2`. Please verify.
        ffn_config = FeedForwardConfigStruct(
            hidden_size=module.fc1.in_features,
            intermediate_size=module.fc1.out_features,
            intermediate_act_type="gelu" # Assuming GELU, please verify
        )
        return cls(
            module=module, config=ffn_config,
            up_projs=[module.fc1], down_projs=[module.fc2],
            up_proj_rnames=["fc1"], down_proj_rnames=["fc2"],
            experts=[module], experts_rname="", moe_gate=None, moe_gate_rname="",
            **kwargs
        )


class InfinityTransformerBlockStruct(TransformerBlockStruct):
    """A specific TransformerBlockStruct for a single Infinity block."""
    attn_struct_cls = InfinityAttentionStruct
    ffn_struct_cls = InfinityFFNStruct


class InfinityTransformerBlockStruct(BaseModuleStruct):
    """
    Represents a single transformer block in the Infinity model.
    Inherits from BaseModuleStruct to get automatic naming and a standard interface.
    """
    def __post_init__(self):
        super().__post_init__()  # This call handles name/key generation
        self.key_modules = self._discover_modules()

    def _discover_modules(self):
        """Finds all targetable nn.Linear layers within this specific block."""
        discovered_modules = {}
        
        # Define the potential submodules and their prefixes to check for
        submodule_map = {
            'sa': getattr(self.module, 'sa', None),    # Self-Attention
            'ca': getattr(self.module, 'ca', None),    # Cross-Attention
            'ffn': getattr(self.module, 'ffn', None),  # Feed-Forward Network
        }

        for submodule_name, submodule in submodule_map.items():
            if submodule:
                # Iterate through layers within the submodule (e.g., sa, ca, ffn)
                for layer_name, mod in submodule.named_modules():
                    if isinstance(mod, nn.Linear):
                        # Construct hierarchical keys and names
                        # Example: self.key = "model_blocks_0"
                        #          submodule_name = "sa"
                        #          layer_name = "q_proj"
                        # -> full_key = "model_blocks_0_sa_q_proj"
                        block_submodule_key = join_name(self.key, submodule_name, sep='_')
                        full_key = join_name(block_submodule_key, layer_name.replace('.', '_'), sep='_')
                        
                        block_submodule_name = join_name(self.name, submodule_name)
                        full_name = join_name(block_submodule_name, layer_name)
                        
                        # Find the immediate parent module and the attribute name for this layer
                        parent_path, _, child_name = layer_name.rpartition('.')
                        parent_module = submodule.get_submodule(parent_path) if parent_path else submodule
                        
                        discovered_modules[full_key] = (full_name, mod, parent_module, child_name)
        
        return discovered_modules

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, "BaseModuleStruct", str], None, None]:
        """Implementation of the abstract method to yield all layers in this block."""
        for key, (full_name, module, parent_module, field_name) in self.key_modules.items():
            yield key, full_name, module, parent_module, field_name


class InfinityModelStruct(BaseTransformerStruct):
    """
    The main hierarchical wrapper for the entire Infinity model, inheriting from BaseTransformerStruct.
    """
    block_struct_cls = InfinityTransformerBlockStruct

    def __post_init__(self):
        # Call the parent's __post_init__ to set up self.name, self.key, etc.
        super().__post_init__()
        from deepcompressor.utils.common import join_name

        # --- NEW LOGIC TO BREAK RECURSIVE DEPENDENCY ---
        # 1. First, pre-calculate the list of block names. This creates the
        #    `self.blocks_names` attribute that child blocks will need to find on the parent
        #    during their own initialization.
        self.blocks_names = [
            join_name(self.name, f"blocks.{i}") for i in range(len(self.module.unregistered_blocks))
        ]
        
        # 2. Now that `self.blocks_names` exists, create the list of block struct objects.
        self.blocks = self._discover_blocks()
        
    # --- Implementation of the 3 abstract properties from BaseTransformerStruct ---
    @property
    def num_blocks(self) -> int:
        return len(self.blocks)
    
    @property
    def block_structs(self) -> list[TransformerBlockStruct]:
        return self.blocks
        
    @property
    def block_names(self) -> list[str]:
        # This property now returns the pre-computed list, satisfying the abstract class requirement.
        return self.blocks_names
        
    def _discover_blocks(self):
        """Creates a structured list of InfinityTransformerBlockStruct objects."""
        block_structs = []
        blocks_to_iterate = self.module.unregistered_blocks
        for i, block in enumerate(blocks_to_iterate):
            # When each child struct is created below, its __post_init__ will now
            # be able to find `self.parent.blocks_names` successfully.
            block_structs.append(
                self.block_struct_cls(
                    module=block, parent=self, fname="blocks",
                    rname=f"blocks.{i}", rkey=f"blocks_{i}", idx=i
                )
            )
        return block_structs

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, "BaseModuleStruct", str], None, None]:
        """Iterator for ALL targetable layers in the entire model."""
        # The super() call correctly iterates through proj_in, all blocks, and proj_out.
        yield from super().named_key_modules()

        # Yield any other top-level layers that are not standard proj_in/proj_out.
        if hasattr(self.module, 'shared_ada_lin'):
            key = join_name(self.key, 'shared_ada_lin', sep='_')
            name = join_name(self.name, 'shared_ada_lin')
            yield key, name, self.module.shared_ada_lin, self, 'shared_ada_lin'


# --- Main execution part of the script ---
if __name__ == "__main__":
  
    # --- Step 1: Set up arguments and load your pretrained Infinity model ---
    print("Setting up arguments and loading Infinity model...")

    # Use the exact arguments from your script
    args = argparse.Namespace(
        pn='1M',
        model_path='/workspace/Infinity/weights/infinity_2b_reg.pth',
        vae_path='/workspace/Infinity/weights/infinity_vae_d32reg.pth',
        text_encoder_ckpt='/workspace/Infinity/weights/flan-t5-xl',
        model_type='infinity_2b',
        vae_type=32,
        text_channels=2048,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        apply_spatial_patchify=0,
        # Adding other args from your script that might be needed by load functions
        cfg_insertion_layer=0,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type='torch',
        seed=0,
        bf16=1,
        save_file='tmp.jpg',
        enable_model_cache=0
    )

    # Load the VAE and the main Infinity transformer model
    vae = load_visual_tokenizer(args)
    infinity_model = load_transformer(vae, args)
    
    print("✅ Successfully loaded Infinity model.")

    print("\n--- Testing Hierarchical InfinityModelStruct with Official Base Classes ---")

    # Instantiate the top-level struct, passing proj_in/out as expected by BaseTransformerStruct
    infinity_struct = InfinityModelStruct(
        module=infinity_model, rname="", rkey="",
        norm_in=getattr(infinity_model, 'norm0_ve', None), norm_in_rname='norm0_ve',
        proj_in=getattr(infinity_model, 'word_embed', None), proj_in_rname='word_embed',
        norm_out=getattr(infinity_model, 'head_nm', None), norm_out_rname='head_nm',
        proj_out=getattr(infinity_model, 'head', None), proj_out_rname='head'
    )

    print("\nIterating through discovered layers via new struct:")
    for key, module_name, module, parent, field_name in infinity_struct.named_key_modules():
        print(f"  - Discovered: key='{key}', name='{module_name}', type='{type(module).__name__}'")