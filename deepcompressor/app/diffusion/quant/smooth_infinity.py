# deepcompressor/app/diffusion/quant/smooth_infinity.py

import torch
import torch.nn as nn
from tqdm import tqdm
import typing as tp

from deepcompressor.app.diffusion.config import DiffusionQuantConfig
from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct, DiffusionBlockStruct

# --- Crucially, we import the original layer-smoothing function ---
# We are REUSING the core logic from the original framework.
from .smooth import smooth_diffusion_layer 

# Import the correct class name from our custom loader
from ..dataset.infinity_calib_loader import InfinityCalibManager
from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct

@torch.inference_mode()
def smooth_infinity_model(
    model: InfinityStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Performs activation-aware smoothing (e.g., SmoothQuant) on the Infinity model.

    This function replaces the default `smooth_diffusion` function. It uses our
    custom `InfinityCalibManager` to handle the stateful, autoregressive nature
    of the model, while reusing the core `smooth_diffusion_layer` logic from the
    original framework for maximum compatibility and robustness.

    Args:
        model (`nn.Module`): The patched Infinity model.
        config (`DiffusionQuantConfig`): The PTQ configuration.
        smooth_cache (`dict[str, torch.Tensor]`, *optional*): Pre-computed smoothing scales.

    Returns:
        `dict`: A dictionary containing the computed smoothing scales for each layer.
    """
    print("--- Starting Infinity-Aware Smoothing Process ---")
    
    # 1. Instantiate our custom data manager.
    print("Initializing Infinity-aware calibration manager...")
    calib_manager = InfinityCalibManager(
        model=model,
        cache_dir=config.calib.path, # Path to your stateful cache
        batch_size=config.calib.batch_size
    )

    # 2. Initialize the smoothing cache.
    smooth_cache = smooth_cache or {}
    
    # 3. Iterate through the layers using our custom generator.
    data_iterator = calib_manager.iter_for_quantization()
    
    # --- IMPROVEMENT: Calculate the total number of items more accurately ---
    total_iterations = len(list(model.iter_transformer_block_structs())) * len(calib_manager.dataset.histories_by_prompt)

    with tqdm(total=total_iterations, desc="Smoothing Infinity Layers") as pbar:
        # The data_iterator now yields one item per block per prompt/scale combination.
        for block_name, (block_struct, block_cache, block_kwargs) in data_iterator:
            
            # This check is now correct because our loader yields block structs.
            if not isinstance(block_struct, DiffusionBlockStruct):
                pbar.update(1)
                continue
            
            # 4. Apply the original smoothing algorithm to the layer.
            #    We are reusing the core logic from `smooth.py`. It receives the
            #    `block_cache` that our custom loader has perfectly prepared.
            smooth_diffusion_layer(
                layer=block_struct,
                config=config,
                smooth_cache=smooth_cache,
                layer_cache=block_cache,
                layer_kwargs=block_kwargs,
            )
            pbar.update(1)

    print("\n✅ Infinity-Aware Smoothing complete.")
    return smooth_cache
