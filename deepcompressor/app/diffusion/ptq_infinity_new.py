# /workspace/deepcompressor/app/var/ptq_infinity.py
#
# Description:
# This script orchestrates the Post-Training Quantization (PTQ) of the Infinity
# model using the SVDQuant algorithm from the deepcompressor library.
# It uses the pre-collected calibration data to apply smoothing and SVD-based
# weight quantization to the model's transformer blocks.
#
import torch
import torch.nn as nn
import os
import argparse
import functools
import typing as tp
from tqdm import tqdm
import sys
import gc
import traceback
import pprint
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

# --- Add paths to your projects ---
sys.path.append('.') 
sys.path.append('/workspace/deepcompressor/Infinity_rep/') 

# --- DeepCompressor and Infinity Imports ---
from deepcompressor.app.diffusion.nn.struct_infinity import InfinityModelStruct

from deepcompressor.app.diffusion.quant import (
    smooth_diffusion,
    quantize_diffusion_weights,
    quantize_diffusion_activations,
)
from deepcompressor.app.diffusion.config import (
    DiffusionPipelineConfig,
    DiffusionPtqRunConfig, 
    DiffusionQuantCacheConfig, 
    DiffusionQuantConfig
)
from deepcompressor.utils import tools
from omniconfig import configclass, ConfigParser
from deepcompressor.data.utils.dtype import eval_dtype

from deepcompressor.app.llm.nn.patch import patch_attention, patch_gemma_rms_norm
from deepcompressor.app.llm.ptq import ptq as llm_ptq

# Your model loading utilities
from Infinity_rep.infinity.models.infinity import Infinity
from Infinity_rep.tools.run_infinity import load_visual_tokenizer, load_transformer

from .config import DiffusionPtqCacheConfig, DiffusionPtqRunConfig, DiffusionQuantCacheConfig, DiffusionQuantConfig

from .quant import (
    load_diffusion_weights_state_dict,
    quantize_diffusion_activations,
    quantize_diffusion_weights,
    rotate_diffusion,
    smooth_diffusion,
)

class InfinityPipeline:
    """A simple container class to mimic the structure of a diffusers pipeline."""
    def __init__(self, transformer, vae, text_encoder=None, text_tokenizer=None):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.device = transformer.device
        self.dtype = transformer.dtype
        self.unet = None # for compatibility

def _infinity_build_factory(
    name: str, path: str, dtype: str | torch.dtype, device: str | torch.device, shift_activations: bool, **kwargs
) -> InfinityPipeline:
    """A factory function to load and construct the full Infinity model pipeline."""
    logger = tools.logging.getLogger(__name__)
    logger.info(f"Building custom Infinity pipeline for model: {name}")
    
    full_args = kwargs['build_kwargs'].get('config_namespace')
    if full_args is None:
        raise ValueError("config_namespace not found in kwargs for _infinity_build_factory")
    
    vae = load_visual_tokenizer(full_args)
    transformer = load_transformer(vae, full_args).to(device).eval()
    
    return InfinityPipeline(transformer, vae)

class InfinityPipelineConfig(DiffusionPipelineConfig):
    """Inherits from DiffusionPipelineConfig and overrides the build method."""
    # Define all fields from the YAML here so omniconfig can parse them.
    model_type: str = "infinity_2b"
    vae_path: str = ""
    pn: str = "1M"
    model_path: str = ""
    text_encoder_ckpt: str = ""
    vae_type: int = 32
    text_channels: int = 2048
    add_lvl_embeding_only_first_block: int = 1
    use_bit_label: int = 1
    rope2d_each_sa_layer: int = 1
    rope2d_normalized_by_hw: int = 2
    apply_spatial_patchify: int = 0
    cfg_insertion_layer: int = 0
    use_scale_schedule_embedding: int = 0
    sampling_per_bits: int = 1
    h_div_w_template: float = 1.0
    use_flex_attn: int = 0
    cache_dir: str = '/dev/shm'
    checkpoint_type: str = 'torch'
    seed: int = 0
    bf16: int = 1
    save_file: str = 'tmp.jpg'
    enable_model_cache: int = 0
    family: str = "infinity"

    def build(self, dtype: torch.dtype | None = None, device: str | torch.device | None = None) -> InfinityPipeline:
        """
        Builds the Infinity pipeline. This override ensures that all custom
        arguments from this config object are correctly passed to the factory.
        """
        if dtype is None: dtype = self.dtype
        if device is None: device = self.device
        
        _factory = self._pipeline_factories.get(self.name)
        if not _factory:
            raise ValueError(f"No pipeline factory registered for the name '{self.name}'")
        
        # This correctly passes all attributes from the YAML file (via vars(self))
        # to the factory as keyword arguments.
        return _factory(**vars(self))

# Register our factory for the name that will be in the YAML file.
DiffusionPipelineConfig.register_pipeline_factory("infinity_2b", _infinity_build_factory)

@configclass
@dataclass
class InfinityPtqRunConfig(DiffusionPtqRunConfig):
    """Top-level configuration class for our script."""
    pipeline: InfinityPipelineConfig

# --- Custom Calibration Iterator and PTQ function (implementations unchanged) ---
def infinity_calibration_iterator(
    model_struct: InfinityModelStruct, 
    calib_cache_path: str,
    batch_size: int,
    device: str = "cuda"
) -> tp.Generator[tuple[str, str, dict[str, torch.Tensor]], None, None]:
    # This function implementation remains the same.
    logger = tools.logging.getLogger(__name__)
    if not os.path.isdir(calib_cache_path):
        raise FileNotFoundError(f"Calibration cache directory not found: {calib_cache_path}")
    cache_files = [os.path.join(calib_cache_path, f) for f in os.listdir(calib_cache_path) if f.endswith('.pt')]
    if not cache_files:
        raise FileNotFoundError(f"No .pt files found in {calib_cache_path}. Please run Step 2 first.")
    
    logger.info(f"Found {len(cache_files)} calibration files to iterate through.")
    
    for f_path in tqdm(cache_files, desc="Calibration Iterator", leave=False):
        try:
            cache_item = torch.load(f_path, map_location=device)
            yield "model_forward", "inputs", {
                "x_BLC_wo_prefix": cache_item.get("input_args")[0],
                **cache_item.get("input_kwargs", {})
            }
        except Exception as e:
            logger.warning(f"Could not load or process cache file {f_path}: {e}")
            continue

def ptq(
    model: InfinityModelStruct, config: DiffusionQuantConfig, cache: DiffusionQuantCacheConfig | None = None,
    load_dirpath: str = "", save_dirpath: str = "", copy_on_save: bool = False, save_model: bool = False,
) -> InfinityModelStruct:
    logger = tools.logging.getLogger(__name__)
    if not isinstance(model, InfinityModelStruct):
        model = InfinityModelStruct.construct(model)
    
    # Run smoothing
    if config.enabled_smooth:
        logger.info("* Smoothing Infinity model for quantization...")
        calibration_iterator = infinity_calibration_iterator(model, config.calib.path, config.calib.batch_size, model.module.device)
        smooth_diffusion(model, config, calibration_iterator=calibration_iterator)
        gc.collect(); torch.cuda.empty_cache()

    # Run weight quantization
    if config.enabled_wgts:
        logger.info("* Quantizing Infinity weights...")
        calibration_iterator = infinity_calibration_iterator(model, config.calib.path, config.calib.batch_size, model.module.device)
        quantize_diffusion_weights(model, config, calibration_iterator=calibration_iterator)
        gc.collect(); torch.cuda.empty_cache()

    # Run activation quantization
    if config.enabled_ipts or config.enabled_opts:
        logger.info("* Quantizing Infinity activations...")
        calibration_iterator = infinity_calibration_iterator(model, config.calib.path, config.calib.batch_size, model.module.device)
        quantize_diffusion_activations(model, config, calibration_iterator=calibration_iterator)
        gc.collect(); torch.cuda.empty_cache()
        
    return model

def main(config: InfinityPtqRunConfig, logging_level: int = tools.logging.DEBUG):
    """Main orchestration script for Infinity PTQ."""
    config.output.lock()
    config.dump(path=config.output.get_running_job_path("config.yaml"))
    tools.logging.setup(path=config.output.get_running_job_path("run.log"), level=logging_level)
    logger = tools.logging.getLogger(__name__)

    logger.info("=== Configurations ===")
    logger.info(pprint.pformat(config.dump(), indent=2, width=120))
    logger.info("=== Output Directory ===")
    logger.info(config.output.job_dirpath)

    logger.info("=== Start Quantization ===")
    logger.info("* Building Infinity model pipeline...")
    
    ### FIX: Pass the config namespace via the standard `build_kwargs` attribute ###
    # This is the correct way to pass extra arguments to a custom factory.
    config.pipeline.build_kwargs = {"config_namespace": argparse.Namespace(**vars(config.pipeline))}
    pipeline = config.pipeline.build()
    logger.info("✅ Pipeline built.")

    model = InfinityModelStruct.construct(pipeline)
    logger.info("✅ Model struct created.")
    
    save_dirpath = ""
    if config.save_model:
        if config.save_model.lower() in ("true", "default"):
            save_dirpath = os.path.join(config.output.running_job_dirpath, "model")
        else:
            save_dirpath = config.save_model
    
    model = ptq(
        model,
        config.quant,
        cache=config.cache,
        load_dirpath=config.load_from,
        save_dirpath=save_dirpath,
        copy_on_save=config.copy_on_save,
        save_model=bool(save_dirpath),
    )
    
    logger.info("✅ Quantization process finished.")
    config.output.unlock()


if __name__ == "__main__":
    DiffusionQuantConfig.set_key_map(InfinityModelStruct._get_default_key_map())
    parser = InfinityPtqRunConfig.get_parser()
    config, _, unused_cfgs, unused_args, unknown_args = parser.parse_known_args()
    assert isinstance(config, InfinityPtqRunConfig)
    
    if len(unused_cfgs) > 0: tools.logging.warning(f"Unused configurations: {unused_cfgs}")
    if unused_args is not None: tools.logging.warning(f"Unused arguments: {unused_args}")
    assert len(unknown_args) == 0, f"Unknown arguments: {unknown_args}"
    
    try:
        main(config, logging_level=tools.logging.DEBUG)
    except Exception as e:
        tools.logging.Formatter.indent_reset()
        tools.logging.error("=== Error ===")
        tools.logging.error(traceback.format_exc())
        tools.logging.shutdown()
        traceback.print_exc()
        if hasattr(config, 'output'):
             config.output.unlock(error=True)
        raise e