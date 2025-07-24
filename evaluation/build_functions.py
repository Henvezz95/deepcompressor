
import gc

from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct
from deepcompressor.calib.smooth import ActivationSmoother
from deepcompressor.quantizer import Quantizer
from deepcompressor.utils.hooks import SimpleInputPackager
from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.nn.patch.lowrank import LowRankBranch

def attach_low_rank_branches(model_struct: InfinityStruct, config: DiffusionPtqRunConfig, branch_state_dict: dict):
    """
    Attaches pre-computed low-rank branches with a refactored, clearer logic.
    It handles individual, shared (SA QKV), and partially shared (CA KV) branches correctly.
    """
    print("--- Attaching Low-Rank Branches (W4) ---")
    device = next(model_struct.module.parameters()).device
    dtype = next(model_struct.module.parameters()).dtype
    for module_key, module_name, module, _, _ in model_struct.named_key_modules():
        if len(module_name.split('.')) < 2:
            print(f"Skipping module {module_name}.")
            continue
        if module_name.split('.')[-2] in ['sa', 'ca']:
            if module_name.split('.')[-1] == 'to_q':
                branch = LowRankBranch(
                    in_features=module.in_features, out_features=module.out_features,
                    rank=config.quant.wgts.low_rank.rank
                ).to(device=device, dtype=dtype)
                branch.a.weight.data.copy_(branch_state_dict[module_name]['a.weight'].data)
                b_slice = branch_state_dict[module_name]['b.weight'].data[:2048]
                branch.b.weight.data.copy_(b_slice)
                branch.as_hook().register(module)
            elif module_name.split('.')[-1] == 'to_k':
                q_name = ".".join(module_name.split('.')[:-1]) + '.to_q'
                branch = LowRankBranch(
                    in_features=module.in_features, out_features=module.out_features,
                    rank=config.quant.wgts.low_rank.rank
                ).to(device=device, dtype=dtype)
                branch.a.weight.data.copy_(branch_state_dict[q_name]['a.weight'].data)
                b_slice = branch_state_dict[q_name]['b.weight'].data[2048:4096]
                branch.b.weight.data.copy_(b_slice)
                branch.as_hook().register(module)
            elif module_name.split('.')[-1] == 'to_v':
                q_name = ".".join(module_name.split('.')[:-1]) + '.to_q'
                branch = LowRankBranch(
                    in_features=module.in_features, out_features=module.out_features,
                    rank=config.quant.wgts.low_rank.rank
                ).to(device=device, dtype=dtype)
                branch.a.weight.data.copy_(branch_state_dict[q_name]['a.weight'].data)
                b_slice = branch_state_dict[q_name]['b.weight'].data[4096:]
                branch.b.weight.data.copy_(b_slice)
                branch.as_hook().register(module)
            else:
                if module_name in branch_state_dict:
                    branch = LowRankBranch(
                        in_features=module.in_features, out_features=module.out_features,
                        rank=config.quant.wgts.low_rank.rank
                    ).to(device=device, dtype=dtype)
                    branch.load_state_dict(branch_state_dict[module_name])
                    branch.as_hook().register(module)
                else:
                    print(f'{module_name} not in branch_state_dict (sa type), skipping...')
                    continue 
        else:
            if module_name in branch_state_dict:
                branch = LowRankBranch(
                        in_features=module.in_features, out_features=module.out_features,
                        rank=config.quant.wgts.low_rank.rank
                    ).to(device=device, dtype=dtype)
                branch.load_state_dict(branch_state_dict[module_name])
                branch.as_hook().register(module)
            else:
                print(f'{module_name} not in branch_state_dict, skipping...')
                continue
    print("✅ Low-rank branches attached successfully.")

# This is the corrected function, following your rules.
def attach_activation_hooks(model_struct: InfinityStruct, smooth_scales: dict):
    """Attaches ActivationSmoother and Quantizer hooks to parent modules."""
    print("--- Attaching Activation Hooks (Smoother + Quantizer) ---")
    for module_key, module_name, module, _, _ in model_struct.named_key_modules():
        if len(module_name.split('.')) < 2:
            print(f"Skipping module {module_name}.")
            continue
        if module_name.split('.')[-2] == 'sa':
            if module_name.split('.')[-1] in ['to_q', 'to_k', 'to_v']:
                adapted_name = ".".join(module_name.split('.')[:-1])+'.to_q'
                smoother = ActivationSmoother(
                    smooth_scale=smooth_scales[adapted_name],
                    channels_dim=-1,
                    input_packager=SimpleInputPackager()
                )
                # Register as a pre-hook to run before the quantizer
                smoother.as_hook().register(module)
            else:
                smoother = ActivationSmoother(
                    smooth_scale=smooth_scales[module_name],
                    channels_dim=-1,
                    input_packager=SimpleInputPackager()
                )
                # Register as a pre-hook to run before the quantizer
                smoother.as_hook().register(module)
        elif module_name.split('.')[-2] == 'ca':
            if module_name.split('.')[-1] in ['to_k', 'to_v']:
                adapted_name = ".".join(module_name.split('.')[:-1]) + '.kv_smooth_scale'
                smoother = ActivationSmoother(
                    smooth_scale=smooth_scales[adapted_name],
                    channels_dim=-1,
                    input_packager=SimpleInputPackager()
                )
                # Register as a pre-hook to run before the quantizer
                smoother.as_hook().register(module)
            else:
                smoother = ActivationSmoother(
                    smooth_scale=smooth_scales[module_name],
                    channels_dim=-1,
                    input_packager=SimpleInputPackager()
                )
                # Register as a pre-hook to run before the quantizer
                smoother.as_hook().register(module)
        else:
            if module_name in smooth_scales:
                smoother = ActivationSmoother(
                    smooth_scale=smooth_scales[module_name],
                    channels_dim=-1,
                    input_packager=SimpleInputPackager()
                )
                # Register as a pre-hook to run before the quantizer
                smoother.as_hook().register(module)
            else:
                print(f'{module_name} not in smoothing scales dict, skipping...')
                continue

def assemble_model(model_struct: InfinityStruct, 
                   configs: DiffusionPtqRunConfig, 
                   branch_state_dict: dict, 
                   smooth_scales: dict, weights:dict, 
                   quantize_activations = True):
    
    config = configs.quant
    
    attach_activation_hooks(model_struct, smooth_scales)
    attach_low_rank_branches(model_struct, configs, branch_state_dict)

    if quantize_activations:
        for module_key, module_name, module, _, _ in model_struct.named_key_modules():
            # Check if this layer should have its activations quantized
            if config.ipts.is_enabled_for(module_key):
                #if 'ca.to_k' not in module_name and 'ca.to_v' not in module_name:
                quantizer = Quantizer(config.ipts, key=module_name, channels_dim=-1)
                
                # The crucial fix:
                quantizer.input_packager = SimpleInputPackager()
                quantizer.as_hook().register(module)


    # Load the quantized weights
    model_struct.module.load_state_dict(weights)
    del weights
    del branch_state_dict
    del smooth_scales
    gc.collect()