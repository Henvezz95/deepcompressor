# deepcompressor/app/diffusion/quant/smooth_infinity.py

import torch
import torch.nn as nn
from tqdm import tqdm
import typing as tp

# Import necessary components from your project
from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct, DiffusionBlockStruct

# --- Crucially, we import the original layer-smoothing function ---
# We are REUSING the core logic from the original framework.
from .smooth import smooth_diffusion_layer 

# CORRECTED: Import the correct class name from our custom loader
from ..dataset.infinity_calib_loader import InfinityCalibManager

@torch.inference_mode()
def smooth_infinity_model(
    model: nn.Module,
    config: DiffusionPtqRunConfig,
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
        config (`DiffusionPtqRunConfig`): The PTQ configuration.
        smooth_cache (`dict[str, torch.Tensor]`, *optional*): Pre-computed smoothing scales.

    Returns:
        `dict`: A dictionary containing the computed smoothing scales for each layer.
    """
    print("--- Starting Infinity-Aware Smoothing Process ---")
    
    # 1. Instantiate our custom data manager.
    #    This is the key change from the original `smooth_diffusion` function.
    print("Initializing Infinity-aware calibration manager...")
    # CORRECTED: Instantiate the correct class
    calib_manager = InfinityCalibManager(
        model=model,
        cache_dir=config.calib.path, # Path to your stateful cache
        batch_size=config.calib.batch_size
    )

    # 2. Initialize the smoothing cache.
    #    This dictionary will store the computed smoothing factors.
    smooth_cache = smooth_cache or {}
    
    # 3. Iterate through the layers using our custom generator.
    #    Our custom manager provides the correctly prepared, stateful data.
    data_iterator = calib_manager.iter_for_quantization()
    
    # Get the total number of items for the progress bar
    # This logic may need refinement based on the exact structure of the data manager
    total_iterations = 0
    if hasattr(calib_manager, 'dataset') and hasattr(calib_manager.dataset, 'data_by_layer_and_scale'):
        for block in calib_manager.model_struct.iter_transformer_block_structs():
            num_scales = len(calib_manager.dataset.data_by_layer_and_scale.get(f"{block.rname}.sa", {}))
            total_iterations += num_scales * len(list(block.iter_submodules()))

    with tqdm(total=total_iterations if total_iterations > 0 else None, desc="Smoothing Infinity Layers") as pbar:
        for layer_name, (layer_struct, layer_cache, layer_kwargs) in data_iterator:
            if not isinstance(layer_struct, DiffusionBlockStruct):
                pbar.update(1)
                continue
            
            # 4. Apply the original smoothing algorithm to the layer.
            #    We are reusing the core logic from `smooth.py`. It receives the
            #    `layer_cache` that our custom loader has perfectly prepared.
            smooth_diffusion_layer(
                layer=layer_struct,
                config=config,
                smooth_cache=smooth_cache,
                layer_cache=layer_cache,
                layer_kwargs=layer_kwargs,
            )
            pbar.update(1)

    print("\n✅ Infinity-Aware Smoothing complete.")
    return smooth_cache
