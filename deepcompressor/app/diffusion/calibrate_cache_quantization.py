from tqdm import tqdm
import torch, json

from deepcompressor.utils import tools
from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct, patchModel
from deepcompressor.app.diffusion.nn.struct import DiTStruct

from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
import math

import os

from .dataset.infinity_calib_loader_new import InfinityCalibManager

# Your model loading utilities
from .dataset.collect.online_infinity_generation import load_transformer, load_visual_tokenizer, args
from deepcompressor.app.diffusion.dataset.collect.build_quantized_infinity_for_calib import load_quantized_model_for_calib

def per_channel_percentiles(tc, low=0.001, high=0.999, use_cpu=True):
    # Standardize so shapes are (T, C, ...)
    xs = tc.inputs.tensors[0].get_standardized_data(reshape=False)  # flattens dims before channels_dim
    xs = [x.detach() for x in xs]                 # drop autograd graphs, keep device
    x = torch.cat(xs, dim=0)                      # (sum_T, C, ...)

    # Compute stats in fp32 (on CPU to spare VRAM), but keep your original tensors on GPU
    dev = torch.device("cpu") if use_cpu else x.device
    x = x.to(dev, dtype=torch.float32, non_blocking=True)

    # Flatten all non-channel dims → (C, N)
    C = x.shape[1]
    if x.ndim == 2:
        x_flat = x.transpose(0, 1)               # (C, T)
    else:
        x_flat = x.permute(1, 0, *range(2, x.ndim)).reshape(C, -1)  # (C, N)

    lo = torch.quantile(x_flat, low, dim=1)      # (C,)
    hi = torch.quantile(x_flat, high, dim=1)     # (C,)
    return lo, hi


# Affine (asymmetric) int8 quantization params per channel
def affine_int8_params(lo, hi, qmin=-128, qmax=127, eps=1e-12):
    # Avoid zero or negative ranges
    scale = (hi - lo).clamp_min(eps) / float(qmax - qmin)         # (C,)
    zero_point = (qmin - lo / scale).round().clamp(qmin, qmax)    # (C,)
    return scale, zero_point.to(torch.int32)


# ---- core helpers ----
def flatten_tc_to_CN(tc, use_cpu=True):
    #xs = tc.inputs.tensors[0].get_standardized_data(reshape=False)
    #x = torch.cat(xs, dim=0)
    #x = torch.cat(chunks, dim=0)  # (sum_N_pre, channels[, ...])
    x = torch.cat([torch.cat(tc.inputs.tensors[0].data, dim=2)[0], 
                   torch.cat(tc.inputs.tensors[0].data, dim=2)[1]], dim=1)
    
    dev = torch.device("cpu") if use_cpu else x.device
    x = x.to(dev, dtype=torch.float32, non_blocking=True)
    return x

def per_channel_lo_hi_from_p(x_flat, p):
    """
    x: per-channel tensor with a distinct tokens axis.
       Examples:
         (H, T, Dh)  -> tokens_dim=1
         (H, Dh, N)  -> tokens_dim=2
         (C, N)      -> tokens_dim=1
    returns: lo, hi with the same channel shape as x minus the tokens axis.
    """
    qs = torch.tensor([1.0 - p, p], dtype=x_flat.dtype, device=x_flat.device)
    q = torch.quantile(x_flat, qs, dim=1)
    return q[0], q[1]  # lo, hi

def qdq_mse(x_flat, lo, hi, qmin=-128, qmax=127):
    scale = (hi - lo).clamp_min(1e-12) / float(qmax - qmin)
    zp = (qmin - lo / scale).round().clamp(qmin, qmax)
    q = (x_flat / scale[:, None] + zp[:, None]).round().clamp(qmin, qmax)
    x_hat = (q - zp[:, None]) * scale[:, None]
    mse = ((x_hat - x_flat) ** 2).mean(dim=1)  # (C,)
    return mse, scale, zp.to(torch.int32)

def eval_p(X, p, agg="median", qmin=-128, qmax=127):
    lo, hi = per_channel_lo_hi_from_p(X, p)  # (H, Dh)
    mse, scale, zp = qdq_mse(X, lo, hi, qmin=qmin, qmax=qmax)                   # (H,), (H,Dh), (H,Dh)
    score = mse.median() if agg == "median" else mse.mean()
    return float(score), (lo, hi, mse, scale, zp)

def search_best_p_unimodal(tc, agg="median", use_cpu=True, steps_refine=3,
                           p_low=0.99, p_high=0.9999, qmin=-128, qmax=127):
    """
    Coarse grid + golden-section (unimodal-friendly; no monotonicity assumed).
    """
    x_res = flatten_tc_to_CN(tc, use_cpu=use_cpu)

    # 1) coarse grid
    grid = [0.98, 0.985, 0.99, 0.992, 0.995, 0.997, 0.9985, 0.999, 0.9995, 0.9998, 0.9999, 0.99999]
    grid = [p for p in grid if p_low <= p <= p_high]
    vals = [eval_p(x_res, p, agg=agg, qmin=qmin, qmax=qmax) for p in grid]
    best_idx = min(range(len(vals)), key=lambda i: vals[i][0])
    best_p, best = grid[best_idx], vals[best_idx]
    # bracket for refine
    lo_p = grid[max(0, best_idx - 1)]
    hi_p = grid[min(len(grid) - 1, best_idx + 1)]
    if lo_p == hi_p:
        lo_p, hi_p = max(p_low, best_p - 0.0005), min(p_high, best_p + 0.0005)

    # 2) golden-section on [lo_p, hi_p]
    phi = (1 + math.sqrt(5)) / 2
    invphi = 1 / phi
    a, b = lo_p, hi_p
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi
    fc = eval_p(x_res, c, agg=agg, qmin=qmin, qmax=qmax)
    fd = eval_p(x_res, d, agg=agg, qmin=qmin, qmax=qmax)

    for _ in range(steps_refine):
        if fc[0] < fd[0]:
            b, fd = d, fc
            d = c
            c = b - (b - a) * invphi
            fc = eval_p(x_res, c, agg=agg, qmin=qmin, qmax=qmax)
        else:
            a, fc = c, fd
            c = d
            d = a + (b - a) * invphi
            fd = eval_p(x_res, d, agg=agg, qmin=qmin, qmax=qmax)

    # pick best of endpoints + c/d
    candidates = [(a, *eval_p(x_res, a, agg=agg, qmin=qmin, qmax=qmax)),
                  (c, *fc),
                  (d, *fd),
                  (b, *eval_p(x_res, b, agg=agg, qmin=qmin, qmax=qmax)),
                  (best_p, *best)]
    p_star, score, (lo, hi, mse, scale, zp) = min(candidates, key=lambda t: t[1])

    return {
        "best_p": p_star,
        "best_mse_agg": score,
        "best_lo": lo, "best_hi": hi,
        "mse_per_channel": mse,
        "scale": scale, "zero_point": zp,
        "bracket": (a, b),
    }


def main(config: DiffusionPtqRunConfig, unused_cfgs: dict, logging_level: int = tools.logging.DEBUG, use_quantized: bool = True) -> None:
    """Post-training quantization of a diffusion model.
            The diffusion model post-training quantization configuration.
        logging_level (`int`, *optional*, defaults to `logging.DEBUG`):
            The logging level.

    Returns:
        `DiffusionPipeline`:
            The diffusion pipeline with quantized model.
    """
    save_dir = os.path.join(config.output.root, "kv_scales")
    os.makedirs(save_dir, exist_ok=True)

    if use_quantized:
        # Pick where your artifacts live (same you used in benchmark)
        artifact_dir = "runs/diffusion/int4_rank32_batch12/model/"  # or from config

        # Get quantized model (W4 or W4A depending on flag)
        patched_model = load_quantized_model_for_calib(
            artifact_dir=artifact_dir,
            ptq_config=config,
            use_fake_act=True,    # set False if you want W4-only calibration
            device="cuda"
        )
    else:
        vae = load_visual_tokenizer(args)
        infinity_model = load_transformer(vae, args)
        print("Successfully loaded Infinity model.")
        print("--- Patching attention layers to be compatible ---")
        # Instantiate the top-level struct, passing proj_in/out as expected by BaseTransformerStruct
        patched_model = patchModel(infinity_model)


    # Build the struct the calib loop expects (as before)
    model = DiTStruct.construct(patched_model)

    # Give modules stable names
    for name, module in patched_model.named_modules():
        module.name = name

    # Inference hygiene
    patched_model.eval().requires_grad_(False).to("cuda")
    torch.cuda.empty_cache()
    
    if os.path.exists(os.path.join(save_dir, "kv_quant_calib.pt")):
        calib = torch.load(os.path.join(save_dir, "kv_quant_calib.pt"))
        params = calib["params"]
        percentiles = calib["percentiles"]
        diagnostics = calib["diagnostics"]
        done_keys = set(".".join(p.split('.')[:4]) for p in params.keys())
        done_keys = list(done_keys)
    else:
        params, percentiles, diagnostics = {}, {}, {}
        done_keys = []

    print(' Generating Evaluation Images')
    calib_manager = InfinityCalibManager(
        model = model, 
        config = config, 
        other_configs = unused_cfgs, 
        smooth_cache = {},
        save_kv_cache_only= True,
        save_imgs=False,
        skip_keys=done_keys
    )

    num_blocks = len(list(model.iter_transformer_block_structs()))
    data_iterator = calib_manager.iter_layer_activations()

    with tqdm(total=num_blocks*2, desc="Measuring Cache Ranges per layer") as pbar:
        for _, aggregated_cache, _ in data_iterator:
            for key, tc in aggregated_cache.items():
                if not aggregated_cache:
                    print(f"Skipping {key}, already calibrated")
                    pbar.update(1)
                    continue
                
                # Find best p in [99.0%, 99.999%] with 3 narrowing steps, minimizing median per-channel MSE
                res = search_best_p_unimodal(
                    tc,
                    steps_refine=3,
                    p_low=0.9,
                    p_high=0.999999,
                    agg="median",           
                    use_cpu=True,     # stats on CPU to save VRAM
                    qmin=-128,
                    qmax=127,
                )

                # Chosen per-channel thresholds
                lo, hi = res["best_lo"], res["best_hi"]
                s, zp = res["scale"], res["zero_point"]  # per-channel int8 params

                # (optional) quick log
                print(f"[{key}] best p={res['best_p']:.6f} | agg MSE={res['best_mse_agg']:.6g}")

                # Store results
                percentiles[key] = {"lo": lo, "hi": hi, "p": res["best_p"]}
                params[key] = {"scale": s, "zero_point": zp}
                diagnostics[key] = {
                    "mse_per_channel": res["mse_per_channel"],
                    "mse_agg": res["best_mse_agg"],
                }
                # Save partial results after each block
                torch.save({
                    "params": params,
                    "percentiles": percentiles,
                    "diagnostics": diagnostics,
                }, os.path.join(save_dir, "kv_quant_calib.pt"))

            pbar.update(1)

if __name__ == "__main__":
    config, _, unused_cfgs, unused_args, unknown_args = DiffusionPtqRunConfig.get_parser().parse_known_args()
    main(config, unused_cfgs, logging_level=tools.logging.DEBUG)