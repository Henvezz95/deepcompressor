import torch
import sys
import argparse 
import cv2
import numpy as np
import os
import gc
from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.utils import tools

# Ensure the submodules are discoverable
sys.path.append('./Infinity_rep') 

from Infinity_rep.tools.run_infinity import (
    load_visual_tokenizer, load_transformer, load_tokenizer, 
    gen_one_img, h_div_w_templates, dynamic_resolution_h_w
)
from Infinity_rep.app.diffusion.dataset.collect.online_infinity_generation import args_2b, args_8b
from evaluation.quantized_layers import (
    swap_linear_for_svdquant, load_svdquant_artifacts, 
    load_svdquant_weights, attach_kv_qparams
)

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    # 1. Initialize the global configuration parser
    logger = tools.logging.getLogger(__name__)
    ptq_config, _, _, _, unknown_args = DiffusionPtqRunConfig.get_parser().parse_known_args()
    ptq_config.output.lock()

    # 2. Define custom arguments (Matching benchmark_assembled_model pattern)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, required=True, help="Path to PTQ artifacts (model.pt, etc.)")
    parser.add_argument("--enable_kv_quant", type=str2bool, default=True, help="Enable KV cache quantization")
    parser.add_argument("--prompt", type=str, default="A photo of a happy dog")
    parser.add_argument("--seed", type=int, default=16)
    custom_args, _ = parser.parse_known_args(unknown_args)

    # 3. Resolve Model-Specific Architecture
    if ptq_config.pipeline.name == 'infinity_2b':
        args = args_2b
        cache_path = os.path.join('runs', "kv_scales", "kv_quant_calib.pt")
    elif ptq_config.pipeline.name == 'infinity_8b':
        args = args_8b
        cache_path = os.path.join('runs', "kv_scales", "kv_quant_calib_8b.pt")
    else:
        raise NotImplementedError(f"Pipeline {ptq_config.pipeline.name} not implemented")

    # Update Infinity internal args with YAML config values
    args.model_path = ptq_config.pipeline.model_path
    args.vae_path = ptq_config.pipeline.vae_path
    args.text_encoder_ckpt = ptq_config.pipeline.text_encoder_ckpt

    # 4. Model Loading
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    model = load_transformer(vae, args).eval()

    # Setup Schedule
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 1.0))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    # 5. Transform to Real Quantized Architecture
    for name, module in model.named_modules():
        module.name = name
    
    if custom_args.enable_kv_quant:
        attach_kv_qparams(model, cache_path)

    # Swap to Nunchaku-optimized kernels
    swap_linear_for_svdquant(model, lora_rank=32, exclude_names=[
        'word_embed', 'head', 'text_norm', 'norm0_cond', 
        'text_proj_for_sos', 'text_proj_for_ca', 'lvl_embed', 
        'shared_ada_lin', 'head_nm', 'ca.mat_kv'
    ]) 
    
    # Load Real Weights and SVD Branches
    arts = load_svdquant_artifacts(custom_args.base_path)
    load_svdquant_weights(model, arts, strict=True)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 6. Benchmark Execution
    print(f"--- Benchmarking {ptq_config.pipeline.name} (Real Quantization) ---")
    for i in range(5):
        img = gen_one_img(
            model, vae, text_tokenizer, text_encoder,
            custom_args.prompt, g_seed=custom_args.seed,
            cfg_list=3.0, tau_list=0.5, scale_schedule=scale_schedule, 
            vae_type=args.vae_type, sampling_per_bits=args.sampling_per_bits
        )
        if i == 0: cv2.imwrite('real_quant_output.png', img.detach().cpu().numpy())

    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\n[FINAL REPORT]")
    print(f"Peak GPU Memory: {peak_mem:.2f} GB")

if __name__ == "__main__":
    main()
