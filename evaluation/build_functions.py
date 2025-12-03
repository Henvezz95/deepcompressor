
import gc

from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct
from deepcompressor.calib.smooth import ActivationSmoother
from deepcompressor.quantizer import Quantizer
from deepcompressor.utils.hooks import SimpleInputPackager
from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.nn.patch.lowrank import LowRankBranch
import torch

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
                hidden_size = branch_state_dict[module_name]['b.weight'].data.shape[0]//3
                branch = LowRankBranch(
                    in_features=module.in_features, out_features=module.out_features,
                    rank=config.quant.wgts.low_rank.rank
                ).to(device=device, dtype=dtype)
                branch.a.weight.data.copy_(branch_state_dict[module_name]['a.weight'].data)
                b_slice = branch_state_dict[module_name]['b.weight'].data[:hidden_size]
                branch.b.weight.data.copy_(b_slice)
                branch.as_hook().register(module)
            elif module_name.split('.')[-1] == 'to_k':
                q_name = ".".join(module_name.split('.')[:-1]) + '.to_q'
                hidden_size = branch_state_dict[q_name]['b.weight'].data.shape[0]//3
                branch = LowRankBranch(
                    in_features=module.in_features, out_features=module.out_features,
                    rank=config.quant.wgts.low_rank.rank
                ).to(device=device, dtype=dtype)
                branch.a.weight.data.copy_(branch_state_dict[q_name]['a.weight'].data)
                b_slice = branch_state_dict[q_name]['b.weight'].data[hidden_size:hidden_size*2]
                branch.b.weight.data.copy_(b_slice)
                branch.as_hook().register(module)
            elif module_name.split('.')[-1] == 'to_v':
                q_name = ".".join(module_name.split('.')[:-1]) + '.to_q'
                hidden_size = branch_state_dict[q_name]['b.weight'].data.shape[0]//3
                branch = LowRankBranch(
                    in_features=module.in_features, out_features=module.out_features,
                    rank=config.quant.wgts.low_rank.rank
                ).to(device=device, dtype=dtype)
                branch.a.weight.data.copy_(branch_state_dict[q_name]['a.weight'].data)
                b_slice = branch_state_dict[q_name]['b.weight'].data[hidden_size*2:]
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
                   smooth_scales: dict, 
                   weights: dict, 
                   quantize_activations: bool = True,
                   skip_ca_kv_act: bool = False):
    
    config = configs.quant
    
    if smooth_scales:
        attach_activation_hooks(model_struct, smooth_scales)
    if branch_state_dict:
        attach_low_rank_branches(model_struct, configs, branch_state_dict)

    if quantize_activations:
        for module_key, module_name, module, _, _ in model_struct.named_key_modules():
            # Check if this layer should have its activations quantized
            if not config.ipts.is_enabled_for(module_key):
                print(f'Skipping {module_name} activation quantization - 1')
                continue
            # Skips kv projection input quantization in cross attention layers if skip_ca_kv_act is enabled
            if skip_ca_kv_act and ('ca.to_k' in module_name or 'ca.to_v' in module_name):
                print(f'Skipping {module_name} activation quantization - 2')
                continue
            
            quantizer = Quantizer(config.ipts, key=module_name, channels_dim=-1)
            quantizer.input_packager = SimpleInputPackager()
            quantizer.as_hook().register(module)


    # Load the quantized weights
    model_struct.module.load_state_dict(weights)
    del weights
    del branch_state_dict
    del smooth_scales
    gc.collect()

def attach_kv_qparams(model, calib_pt_path, verbose=True):
    blob = torch.load(calib_pt_path, map_location="cpu")
    params = blob["params"]  # dict: key -> {"scale": tensor(C,), "zero_point": tensor(C,)}

    # map module name -> module (requires you assigned module.name = name earlier)
    name2mod = {}
    for m in model.modules():
        n = getattr(m, "name", None)
        if n is not None:
            name2mod[n] = m

    def get_sa(mod):
        # support either .sa or .attn containers
        return getattr(mod, "sa", None) or getattr(mod, "attn", None)

    ok_k = ok_v = 0
    missing_mod = []
    unmatched = []

    for raw_key, pd in params.items():
        # normalize the saved key
        key = str(raw_key).strip()
        if key.startswith("[") and key.endswith("]"):
            key = key[1:-1].strip()

        # locate suffix and derive the module name prefix
        suffix = None
        if ".sa.k.cache" in key:   suffix = ".sa.k.cache"
        elif ".attn.k.cache" in key: suffix = ".attn.k.cache"
        elif ".sa.v.cache" in key: suffix = ".sa.v.cache"
        elif ".attn.v.cache" in key: suffix = ".attn.v.cache"

        if suffix is None:
            unmatched.append(raw_key); 
            continue

        base = key.split(suffix, 1)[0]  # module name prefix
        mod = name2mod.get(base)

        if mod is None:
            missing_mod.append((raw_key, base))
            continue

        sa = get_sa(mod)
        if sa is None:
            missing_mod.append((raw_key, base + " (no .sa/.attn)"))
            continue

        # assign params (create attributes if they don't exist)
        sc = pd["scale"].float()
        zp = pd["zero_point"].to(torch.int32)

        if suffix.endswith("k.cache"):
            sa.k_scale = sc
            sa.k_zp    = zp
            ok_k += 1
        else:
            sa.v_scale = sc
            sa.v_zp    = zp
            ok_v += 1

    # flip kv_quant_enabled only where both K and V are present
    enabled = 0
    for m in name2mod.values():
        sa = get_sa(m)
        if sa is None: 
            continue
        if getattr(sa, "k_scale", None) is not None and getattr(sa, "v_scale", None) is not None:
            sa.kv_quant_enabled = True
            enabled += 1

    if verbose:
        print(f"[attach_kv_qparams] assigned K to {ok_k} modules, V to {ok_v} modules; enabled kv-quant on {enabled} modules.")
        if missing_mod:
            print("  - Missing modules for keys:")
            for raw_key, base in missing_mod[:10]:
                print(f"    · key={raw_key}  -> base='{base}' not found in model.names")
            if len(missing_mod) > 10:
                print(f"    ... and {len(missing_mod)-10} more")
        if unmatched:
            print("  - Unmatched keys (no .sa/.attn k/v suffix):")
            for k in unmatched[:10]:
                print(f"    · {k}")
            if len(unmatched) > 10:
                print(f"    ... and {len(unmatched)-10} more")
