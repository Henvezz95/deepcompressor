import torch
import typing as tp
from tqdm import tqdm

from deepcompressor.app.diffusion.config import DiffusionQuantConfig, DiffusionPtqRunConfig
from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
from deepcompressor.utils import tools

# --- Import your custom data loader ---
from ..dataset.infinity_calib_loader_new import InfinityCalibManager
from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct

# --- Import the core low-rank and weight quantization logic from the framework ---
from .weight import calibrate_diffusion_block_low_rank_branch, quantize_diffusion_block_weights, update_diffusion_block_weight_quantizer_state_dict

@torch.inference_mode()
def quantize_infinity_weights(
    model: InfinityStruct,
    config: DiffusionQuantConfig,
    config_loader: DiffusionPtqRunConfig,
    other_configs: dict,
    quantizer_state_dict: dict | None = None,
    branch_state_dict: dict | None = None,
    return_with_scale_state_dict: bool = False,
) -> tuple[dict, dict, dict | None]:
    """
    An adapted version of quantize_diffusion_weights that uses the custom
    InfinityCalibManager to handle the stateful, variable-shape data needed
    for low-rank branch calibration (SVD).
    """
    if not isinstance(model, DiffusionModelStruct):
        model = DiffusionModelStruct.construct(model)
    assert isinstance(model, DiffusionModelStruct)
    quantizer_state_dict = quantizer_state_dict or {}
    branch_state_dict = branch_state_dict or {}

    logger = tools.logging.getLogger(f"{__name__}.WeightQuant")

    # This part remains the same: add the low-rank branches first.
    if config.wgts.enabled_low_rank and (not config.wgts.low_rank.compensate or config.wgts.low_rank.num_iters > 1):
        logger.info("* Adding low-rank branches to weights")
        
        calib_manager = InfinityCalibManager(
            model = model, 
            config = config_loader, 
            other_configs = other_configs, 
            smooth_cache = {}
        )
        data_iterator = calib_manager.iter_layer_activations()
        num_blocks = len(list(model.iter_transformer_block_structs()))
        logger.info("* Adding low-rank branches to weights")
        tools.logging.Formatter.indent_inc()
        with tools.logging.redirect_tqdm():
            if branch_state_dict:
                #### NO CODE IMPLEMENTATION FOR NOW ####
                pass
            else:
                # Use the custom data loader to calibrate the branches
                with tqdm(total=num_blocks, desc="Calibrating low-rank branches") as pbar:
                    for block_struct, aggregated_cache, block_kwargs in data_iterator:
                        calibrate_diffusion_block_low_rank_branch(
                            layer=block_struct,
                            config=config,
                            branch_state_dict=branch_state_dict,
                            layer_cache=aggregated_cache,
                            layer_kwargs=block_kwargs,
                        )
                        pbar.update(1)
                        break # Just for debugging, remove this after
        tools.logging.Formatter.indent_dec()

    skip_pre_modules = all(key in config.wgts.skips for key in model.get_prev_module_keys())
    skip_post_modules = all(key in config.wgts.skips for key in model.get_post_module_keys())
    with tools.logging.redirect_tqdm():
        if not quantizer_state_dict:
            if config.wgts.needs_calib_data:
                # Code implementation still missing #
                pass
            else:
                iterable = map(  # noqa: C417
                    lambda kv: (kv[0], (kv[1], {}, {})),
                    model.get_named_layers(
                        skip_pre_modules=skip_pre_modules, skip_post_modules=skip_post_modules
                    ).items(),
                )
            for _, (layer, layer_cache, layer_kwargs) in tqdm(
                iterable,
                desc="calibrating weight quantizers",
                leave=False,
                total=model.num_blocks + int(not skip_post_modules) + int(not skip_pre_modules) * 3,
                dynamic_ncols=True,
            ):
                update_diffusion_block_weight_quantizer_state_dict(
                    layer=layer,
                    config=config,
                    quantizer_state_dict=quantizer_state_dict,
                    layer_cache=layer_cache,
                    layer_kwargs=layer_kwargs,
                )
        else:
            # Code implementation still missing #
            pass

    scale_state_dict: dict[str, torch.Tensor | float | None] = {}
    if config.wgts.enabled_gptq:
        # Code implementation still missing #
        pass
    else:
        iterable = map(  # noqa: C417
            lambda kv: (kv[0], (kv[1], {}, {})),
            model.get_named_layers(skip_pre_modules=skip_pre_modules, skip_post_modules=skip_post_modules).items(),
        )

    for _, (layer, layer_cache, _) in tqdm(
        iterable,
        desc="quantizing weights",
        leave=False,
        total=model.num_blocks + int(not skip_post_modules) + int(not skip_pre_modules) * 3,
        dynamic_ncols=True,
    ):
        layer_scale_state_dict = quantize_diffusion_block_weights(
            layer=layer,
            config=config,
            layer_cache=layer_cache,
            quantizer_state_dict=quantizer_state_dict,
            return_with_scale_state_dict=return_with_scale_state_dict,
        )
        scale_state_dict.update(layer_scale_state_dict)
    return quantizer_state_dict, branch_state_dict, scale_state_dict
