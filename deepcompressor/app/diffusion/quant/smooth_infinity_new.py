# deepcompressor/app/diffusion/quant/smooth_infinity.py

import torch
import torch.nn as nn
from tqdm import tqdm
import typing as tp
import math
import datasets
from deepcompressor.utils import tools
from deepcompressor.utils.hooks import SimpleInputPackager, SimpleOutputPackager

from deepcompressor.app.diffusion.config import DiffusionQuantConfig, DiffusionPtqRunConfig
from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct, DiffusionBlockStruct
from deepcompressor.data.cache import IOTensorsCache, TensorsCache, TensorCache
from deepcompressor.calib.smooth import ActivationSmoother, smooth_linear_modules
from deepcompressor.quantizer import Quantizer

from ..nn.struct import (
    DiffusionAttentionStruct,
    DiffusionBlockStruct,
    DiffusionFeedForwardStruct,
    DiffusionModelStruct,
    DiffusionTransformerBlockStruct,
)

# --- We reuse the core smoothing logic from the original framework ---
from .smooth import smooth_diffusion_layer, smooth_diffusion_qkv_proj, smooth_diffusion_down_proj, smooth_diffusion_up_proj, smooth_diffusion_out_proj

# Import the final, correct class name from our custom loader
from ..dataset.infinity_calib_loader_new import InfinityCalibManager
from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct

@torch.inference_mode()
def smooth_infinity_model(
    model: InfinityStruct,
    config_loader: DiffusionPtqRunConfig,
    other_configs: dict,
    smooth_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Performs activation-aware smoothing on the Infinity model. This version is
    fully compatible with the lazy-loading, stateful InfinityCalibManager.
    """
    print("--- Starting Infinity-Aware Smoothing Process ---")
    config = config_loader.quant
    smooth_cache = smooth_cache or {}
    if config.smooth.enabled_proj:
        if smooth_cache:
            assert smooth_cache.get("proj.fuse_when_possible", True) == config.smooth.proj.fuse_when_possible
    if config.smooth.enabled_attn:
        if smooth_cache:
            assert smooth_cache.get("attn.fuse_when_possible", True) == config.smooth.attn.fuse_when_possible
    
    # 1. Instantiate our custom data manager.
    print("Initializing Infinity-aware calibration manager...")
    print(config.calib.path+'/caches/')
    calib_manager = InfinityCalibManager(
        model = model, 
        config = config_loader, 
        other_configs = other_configs, 
        smooth_cache = smooth_cache
    )
    
    # 2. Initialize the smoothing cache.
    smooth_cache = smooth_cache or {}
    
    # The total number of iterations is now simply the number of blocks in the model.
    num_blocks = len(list(model.iter_transformer_block_structs()))
    
    # 3. Iterate through the layers using the new, correct generator.
    data_iterator = calib_manager.iter_layer_activations()
    
    print(f"Beginning smoothing for {num_blocks} transformer blocks...")
    with tqdm(total=num_blocks, desc="Smoothing Infinity Blocks") as pbar:
        for block_struct, aggregated_cache, block_kwargs in data_iterator:
            smooth_attention_block(
                layer=block_struct,
                config=config,
                smooth_cache=smooth_cache,
                layer_cache=aggregated_cache,
                layer_kwargs=block_kwargs, # Pass the kwargs containing ca_kv
            )
            pbar.update(1)

    if config.smooth.enabled_proj:
        smooth_cache.setdefault("proj.fuse_when_possible", config.smooth.proj.fuse_when_possible)
    if config.smooth.enabled_attn:
        smooth_cache.setdefault("attn.fuse_when_possible", config.smooth.attn.fuse_when_possible)
    print("\n✅ Infinity-Aware Smoothing complete.")
    return smooth_cache

@torch.inference_mode()
def smooth_attention_block(
    layer: DiffusionBlockStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    layer_cache: dict[str, IOTensorsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
) -> None:
    """Smooth a single Infinity model block.

    Args:
        layer (`DiffusionBlockStruct`):
            The Infinity block.
        config (`DiffusionQuantConfig`):
            The quantization configuration.
        smooth_cache (`dict[str, torch.Tensor]`):
            The smoothing scales cache.
        layer_cache (`dict[str, IOTensorsCache]`, *optional*):
            The layer cache.
        layer_kwargs (`list[dict[str, tp.Any]]`, *optional*):
            The layer keyword arguments. Required for kv_cache and ca_kv
    """
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    logger.debug("- Smoothing Infinity Block %s", layer.name)
    tools.logging.Formatter.indent_inc()
    layer_cache = layer_cache or {}
    layer_kwargs = layer_kwargs or {}
    # We skip resnets since we currently cannot scale the Swish function
    visited: set[str] = set()
    for module_key, module_name, module, parent, _ in layer.named_key_modules():
        if isinstance(parent, (DiffusionAttentionStruct, DiffusionFeedForwardStruct)):
            block = parent.parent
            assert isinstance(block, DiffusionTransformerBlockStruct)
            if block.name not in visited:
                logger.debug("- Smoothing Transformer Block %s", block.name)
                visited.add(block.name)
                tools.logging.Formatter.indent_inc()
                for attn in block.attn_structs:
                    if attn.name.split('.')[-1] == 'sa':
                        continue
                        smooth_cache = smooth_diffusion_qkv_proj(
                            attn=attn, config=config, smooth_cache=smooth_cache, block_cache=layer_cache, block_kwargs=layer_kwargs
                        )
                        smooth_cache = smooth_diffusion_out_proj(
                            attn=attn, config=config, smooth_cache=smooth_cache, block_cache=layer_cache, block_kwargs=layer_kwargs
                        )
                    elif attn.name.split('.')[-1] == 'ca':
                        # This is the custom implementation for the Cross-Attention block.
                        logger.debug("- Custom Smoothing for Cross-Attention Block: %s", attn.name)
                        tools.logging.Formatter.indent_inc()

                        # --- Step 1: Smooth the Query Projection (to_q) ---
                        # This is treated as a standalone module because its input (hidden_states) is unique.
                        q_proj_key = attn.qkv_proj_key 
                        q_proj_name = attn.q_proj_name
                        logger.debug("- Smoothing Query Projection: %s", q_proj_name)
                        if q_proj_name not in smooth_cache:
                            q_smooth_scale = smooth_linear_modules(
                                prevs=None,  # No fusible preceding layer
                                modules=[attn.q_proj],
                                scale=smooth_cache.get(q_proj_name, None),
                                config=config.smooth.proj,
                                weight_quantizer=Quantizer(config.wgts, key=q_proj_key),
                                input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=q_proj_key),
                                inputs=layer_cache[q_proj_name].inputs,
                                eval_inputs=layer_cache[attn.name].inputs,
                                eval_module=attn,
                                eval_kwargs=layer_kwargs,
                                develop_dtype=config.develop_dtype,
                            )
                            smooth_cache[q_proj_name] = q_smooth_scale
                        else:
                            q_smooth_scale = smooth_cache[q_proj_name]
                        # Since prevs=None, we register a hook to scale the activation at runtime.
                        # Do this (replace 'DefaultInputPackager' with whatever you find):
                        smoother = ActivationSmoother(q_smooth_scale, channels_dim=-1)
                        smoother.input_packager = SimpleInputPackager()  # Use the actual class name
                        smoother.as_hook().register(attn.q_proj)
                        continue
                        # --- Step 2: Smooth the Key and Value Projections (to_k, to_v) together ---
                        # These are grouped because they share the same input (kv_compact from the text prompt).
                        kv_proj_key = attn.qkv_proj_key  # Use 'k' as the representative key
                        k_proj_name = attn.k_proj_name
                        v_proj_name = attn.v_proj_name
                        logger.debug("- Smoothing Key/Value Projections: %s & %s", k_proj_name, v_proj_name)
                        
                        kv_smooth_scale = smooth_linear_modules(
                            prevs=None,  # No fusible preceding layer
                            modules=[attn.k_proj, attn.v_proj],
                            scale=smooth_cache.get(k_proj_name, None),
                            config=config.smooth.proj,
                            weight_quantizer=Quantizer(config.wgts, key=kv_proj_key),
                            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=kv_proj_key),
                            inputs=layer_cache[k_proj_name].inputs, # Use input data for the 'k' projection
                            eval_inputs=layer_cache[attn.name].inputs,
                            eval_module=attn,
                            eval_kwargs=layer_kwargs,
                            develop_dtype=config.develop_dtype,
                        )
                        # Store the scale with a unique key for this block's hook.
                        kv_scale_key = f"{attn.name}.kv_smooth_scale"
                        smooth_cache[kv_scale_key] = kv_smooth_scale
                        # Add the hooks
                        smoother = ActivationSmoother(kv_smooth_scale, channels_dim=-1)
                        smoother.input_packager = SimpleInputPackager()  # Use the actual class name
                        smoother.as_hook().register(attn.k_proj)
                        smoother.as_hook().register(attn.v_proj)

                        # --- Step 5: Handle the output projection ---
                        # The original function for this is correct as it's a standalone layer.
                        continue
                        smooth_cache = smooth_diffusion_out_proj(
                            attn=attn, config=config, smooth_cache=smooth_cache, block_cache=layer_cache, block_kwargs=layer_kwargs
                        )

                        tools.logging.Formatter.indent_dec()

                if block.ffn_struct is not None:
                    continue
                    smooth_cache = smooth_diffusion_up_proj(
                        pre_ffn_norm=block.pre_ffn_norm,
                        ffn=block.ffn_struct,
                        config=config,
                        smooth_cache=smooth_cache,
                        block_cache=layer_cache,
                    )
                    smooth_cache = smooth_diffusion_down_proj(
                        ffn=block.ffn_struct, config=config, smooth_cache=smooth_cache, block_cache=layer_cache
                    )
                if block.add_ffn_struct is not None:
                    continue
                    smooth_cache = smooth_diffusion_up_proj(
                        pre_ffn_norm=block.pre_add_ffn_norm,
                        ffn=block.add_ffn_struct,
                        config=config,
                        smooth_cache=smooth_cache,
                        block_cache=layer_cache,
                    )
                    smooth_cache = smooth_diffusion_down_proj(
                        ffn=block.add_ffn_struct, config=config, smooth_cache=smooth_cache, block_cache=layer_cache
                    )
                tools.logging.Formatter.indent_dec()