# run_infinity_dev.py

import torch
import torch.nn as nn
import sys
import argparse
import typing as tp

# Add parent directory to path to import our modules
sys.path.append('.') 

# Import necessary components from your Infinity project and deepcompressor
# Note: You must ensure 'run_infinity.py' and 'Infinity' are in the python path
# and that 'run_infinity' contains the 'load_visual_tokenizer' and 'load_transformer' functions.
try:
    from Infinity_rep.tools.run_infinity import load_visual_tokenizer, load_transformer
    from deepcompressor.nn.struct.base import BaseModuleStruct
    from deepcompressor.utils.common import join_name
except ImportError:
    print("ERROR: Could not import 'load_visual_tokenizer' and 'load_transformer'.")
    print("Please ensure 'run_infinity.py' is in the project root or accessible in the Python path.")
    sys.exit(1)


from deepcompressor.utils.common import join_name

# --- FINAL REVISED and CORRECTED Hierarchical Wrapper Classes ---

class InfinityTransformerBlockStruct(BaseModuleStruct):
    """
    Represents a single transformer block in the Infinity model.
    It inherits from BaseModuleStruct to get automatic naming and a standard interface.
    """
    def __post_init__(self):
        super().__post_init__()
        self.key_modules = self._discover_modules()
        
    def _discover_modules(self):
        """Finds all targetable layers within this specific block."""
        discovered_modules = {}
        
        # Define the potential submodules and their prefixes to check for
        submodule_map = {
            'sa': getattr(self.module, 'sa', None),    # Self-Attention
            'ca': getattr(self.module, 'ca', None),    # Cross-Attention
            'ffn': getattr(self.module, 'ffn', None),  # Feed-Forward Network
        }

        for prefix_name, submodule in submodule_map.items():
            if submodule:
                for layer_name, mod in submodule.named_modules():
                    # Target nn.Linear layers, common in 'proj' and 'fc' layers
                    if isinstance(mod, nn.Linear) and ('proj' in layer_name or 'fc' in layer_name):
                        # Construct hierarchical keys and names
                        full_key = join_name(join_name(self.key, prefix_name, sep='_'), layer_name.replace('.', '_'), sep='_')
                        full_name = join_name(join_name(self.name, prefix_name), layer_name)
                        
                        # Find the immediate parent module and the attribute name for this layer
                        parent_path, _, child_name = layer_name.rpartition('.')
                        parent_module = submodule.get_submodule(parent_path) if parent_path else submodule
                        
                        discovered_modules[full_key] = (full_name, mod, parent_module, child_name)
        
        return discovered_modules

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, "BaseModuleStruct", str], None, None]:
        """Implementation of the abstract method to yield all layers in this block."""
        for key, (full_name, module, parent_module, field_name) in self.key_modules.items():
            yield key, full_name, module, parent_module, field_name


class InfinityModelStruct(BaseModuleStruct):
    """
    The main hierarchical wrapper for the Infinity model.
    """
    def __post_init__(self):
        super().__post_init__()
        
        self.top_level_linears = self._discover_top_level_modules()
        self.blocks_names = [join_name(self.name, f"blocks.{i}") for i in range(len(self.module.unregistered_blocks))]
        self.blocks = self._discover_blocks()
        self.num_blocks = len(self.blocks)

    def _discover_top_level_modules(self):
        """Discovers modules that are not inside the main block loop."""
        top_level = {}
        modules_to_check = {
            'word_embed': self.module.word_embed,
            'head': self.module.head,
            'shared_ada_lin': self.module.shared_ada_lin
        }
        for name, mod in modules_to_check.items():
            if isinstance(mod, (nn.Linear, nn.Conv2d)):
                top_level[name] = mod
        return top_level

    def _discover_blocks(self):
        """Creates a structured list of InfinityTransformerBlockStruct objects."""
        block_structs = []
        blocks_to_iterate = self.module.unregistered_blocks
        for i, block in enumerate(blocks_to_iterate):
            block_structs.append(
                InfinityTransformerBlockStruct(
                    module=block, parent=self, fname="blocks",
                    rname=f"blocks.{i}", rkey=f"blocks_{i}", idx=i
                )
            )
        return block_structs

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, "BaseModuleStruct", str], None, None]:
        """Iterator for ALL targetable layers in the entire model."""
        for name, mod in self.top_level_linears.items():
            key = join_name(self.key, name, sep='_')
            full_name = join_name(self.name, name)
            yield key, full_name, mod, self, name
        
        for block_struct in self.blocks:
            yield from block_struct.named_key_modules()


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

    print("\n--- Testing Hierarchical InfinityModelStruct ---")

    # Instantiate the top-level struct.
    # Since it's the root, parent is None and names/keys are empty.
    infinity_struct = InfinityModelStruct(
        module=infinity_model,
        rname="",
        rkey=""
    )

    # ... (your skip_keywords list remains the same) ...

    # The rest of your verification loop can stay the same,
    # it will now use the new hierarchical named_key_modules iterator.
    print("\nDiscovered nn.Linear layers to be quantized (excluding skipped layers):")
    
    # 2. Define keywords for layers to skip (related to AdaLN)
    skip_keywords = [
        "shared_ada_lin", # Generates AdaLN conditioning, potentially sensitive
        "head_nm",        # The adaptive norm before the head
        # Keywords to skip Q/K/V and the first FFN layer, 
        # as they are often preceded by adaptive normalization.
        ".sa.q_proj",
        ".sa.k_proj",
        ".sa.v_proj",
        ".attn.q_proj",
        ".attn.k_proj",
        ".attn.v_proj",
        ".ffn.fc1" 
    ]

    print("\nDiscovered nn.Linear layers to be quantized (excluding skipped layers):")
    
    # 3. Iterate and print discovered layers
    target_layer_count = 0
    skipped_layer_count = 0
    for key, module_name, module, parent, field_name in infinity_struct.named_key_modules():
        is_skipped = any(skip_word in key for skip_word in skip_keywords)
        
        if is_skipped:
            skipped_layer_count += 1
            # You can uncomment the next line to see which layers are being skipped
            # print(f"  - SKIPPED Layer: {key}")
        else:
            print(f"  - TARGET Layer: {key} (Type: {type(module).__name__})")
            target_layer_count += 1

    print(f"\nTotal targetable layers found: {target_layer_count}")
    print(f"Total layers skipped: {skipped_layer_count}")