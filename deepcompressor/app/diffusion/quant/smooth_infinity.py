# deepcompressor/app/diffusion/quant/smooth_infinity.py

import torch
import torch.nn as nn
from tqdm import tqdm
import typing as tp
import math

from deepcompressor.app.diffusion.config import DiffusionQuantConfig
from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct, DiffusionBlockStruct

# --- We reuse the core smoothing logic from the original framework ---
from .smooth import smooth_diffusion_layer 

# Import the final, correct class name from our custom loader
from ..dataset.infinity_calib_loader import InfinityCalibManager
from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct

@torch.inference_mode()
def smooth_infinity_model(
    model: InfinityStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Performs activation-aware smoothing on the Infinity model. This version is
    fully compatible with the lazy-loading, stateful InfinityCalibManager.
    """
    print("--- Starting Infinity-Aware Smoothing Process ---")
    
    # 1. Instantiate our custom data manager.
    print("Initializing Infinity-aware calibration manager...")
    calib_manager = InfinityCalibManager(
        model=model,
        cache_dir=config.calib.path+'/caches/',
        batch_size=config.calib.batch_size
    )

    # 2. Initialize the smoothing cache.
    smooth_cache = smooth_cache or {}
    
    # The total number of iterations is now simply the number of blocks in the model.
    num_blocks = len(list(model.iter_transformer_block_structs()))
    
# 3. Iterate through the layers using the new, correct generator.
    data_iterator = calib_manager.iter_layer_activations()
    
    print(f"Beginning smoothing for {num_blocks} transformer blocks...")
    with tqdm(total=num_blocks, desc="Smoothing Infinity Blocks") as pbar:
        # --- START of FIX ---
        for block_struct, aggregated_cache, block_kwargs in data_iterator:
            
            smooth_diffusion_layer(
                layer=block_struct,
                config=config,
                smooth_cache=smooth_cache,
                layer_cache=aggregated_cache,
                layer_kwargs=block_kwargs, # Pass the kwargs containing ca_kv
            )
            pbar.update(1)


    print("\n✅ Infinity-Aware Smoothing complete.")
    return smooth_cache

