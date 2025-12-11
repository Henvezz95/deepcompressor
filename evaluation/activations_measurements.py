import torch
import numpy as np
from scipy.stats import kurtosis, skew
import json
from tqdm import tqdm
import gc

# Reuse your existing infrastructure
from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.app.diffusion.nn.struct_infinity import  patchModel
from deepcompressor.app.diffusion.nn.struct import DiTStruct

# Import your custom loader (assumed to be in the python path based on your uploads)
from deepcompressor.app.diffusion.dataset.infinity_calib_loader_new import InfinityCalibManager
from deepcompressor.app.diffusion.dataset.collect.online_infinity_generation import load_visual_tokenizer, load_transformer, args_8b, args_2b

GENERATIVE_STEPS = 13

def calculate_range_variation(tensor, axis):
    """
    Calculates the CV of the Dynamic Range (Max - Min) for Asymmetric Quantization.
    """
    reduction_axis = 1 - axis 
    
    # 1. Compute Min and Max along the reduction axis
    # For a tensor [Tokens, Channels]:
    # If checking Channel variation (axis=1), we find min/max across Tokens (dim=0)
    
    # Note: torch.max/min returns (values, indices) tuple
    min_vals = torch.amin(tensor, dim=reduction_axis)
    max_vals = torch.amax(tensor, dim=reduction_axis)
    
    # 2. Compute the Range (This is the 's' in Asymmetric Quant)
    dynamic_ranges = (max_vals - min_vals).float()
    
    # 3. Compute CV
    mean = torch.mean(dynamic_ranges) + 1e-6
    std = torch.std(dynamic_ranges)
    
    cv = std / mean
    return cv.item()


def analyze_symmetry(tensor):
    """
    Computes Per-Channel Skewness and Zero-Point.
    Prevents 'cancellation' where positive-skewed and negative-skewed 
    channels might look symmetric on average.
    """
    if tensor is None or tensor.numel() == 0: return None
    
    # tensor shape: [Batch*Seq, Channels]
    # We want to analyze distribution across Tokens (axis 0) for EACH Channel
    
    # Move to CPU/Numpy
    x_np = tensor.float().cpu().numpy()
    
    # 1. Fisher-Pearson Skewness (Per Channel)
    # axis=0 computes skewness for each column (channel)
    skew_per_channel = skew(x_np, axis=0) 
    avg_abs_skew = np.mean(np.abs(skew_per_channel))
    
    # 2. Ideal Zero-Point (Per Channel)
    # Calculate min/max for each channel
    min_vals = np.min(x_np, axis=0)
    max_vals = np.max(x_np, axis=0)
    
    # Standard INT8 Affine Quantization Math
    q_min, q_max = -128, 127
    scale = (max_vals - min_vals) / (q_max - q_min + 1e-6)
    
    # ZP = q_min - min / scale
    # Avoid div/0 by masking
    mask = scale > 1e-9
    zero_points = np.zeros_like(scale)
    zero_points[mask] = q_min - min_vals[mask] / scale[mask]
    
    avg_abs_zp = np.mean(np.abs(zero_points))
    
    return {
        "skewness": float(avg_abs_skew),       # Average magnitude of skew
        "max_skew": float(np.max(np.abs(skew_per_channel))),
        "ideal_zero_point": float(avg_abs_zp), # Average magnitude of shift
        "global_min": float(np.min(min_vals)),
        "global_max": float(np.max(max_vals))
    }

def analyze_layer(stats_log, layer_name, tensor, layer_type="linear"):
    # Flatten: [Batch*Seq, Dim]
    if tensor.dim() == 3: # [B, N, C]
        tensor = tensor.reshape(-1, tensor.shape[-1])
    
    # 1. Basic Stats for SVDQuant
    flat_np = tensor.float().cpu().numpy().flatten()
    k_val = kurtosis(flat_np)
    max_val = np.max(np.abs(flat_np))
    median_val = np.median(np.abs(flat_np))
    ratio = max_val / (median_val + 1e-6)
    
    entry = {
        "kurtosis": float(k_val),
        "max_median_ratio": float(ratio),
        "max": float(max_val)
    }

    # 2. KV-Cache Specific Analysis (Channel vs Token Dominance)
    # We want to know: Is the variance driven by Channels or Tokens?
    if layer_type == "kv_cache":
        # tensor is [Tokens, Channels]
        cv_channel = calculate_range_variation(tensor, axis=1) # "How much do channels differ?"
        cv_token = calculate_range_variation(tensor, axis=0)   # "How much do tokens differ?"
        symmetry_analysis = analyze_symmetry(tensor)
        
        entry["cv_channel"] = cv_channel
        entry["cv_token"] = cv_token
        entry["dominance"] = "Channel" if cv_channel > cv_token else "Token"
        for k,v in symmetry_analysis.items():
            entry[k] = v

    if layer_name not in stats_log:
        stats_log[layer_name] = []
    stats_log[layer_name].append(entry)

def main(config: DiffusionPtqRunConfig, unused_cfgs: dict):
    print("--- Starting Infinity 2B Diagnostic ---")
    
    # 1. Load Model (FP16 Baseline)
    if config.pipeline.name == 'infinity_2b':
        args = args_2b
        nname = '2b'
    elif config.pipeline.name == 'infinity_8b':
        args = args_8b
        nname = '8b'
    else:
        raise ValueError("Unknown pipeline")

    vae = load_visual_tokenizer(args)
    stateful_model = load_transformer(vae, args)
    patched_model = patchModel(stateful_model)
    
    model = DiTStruct.construct(patched_model)
    patched_model.eval().requires_grad_(False).to("cuda")

    # 2. Initialize Manager
    # IMPORTANT: save_kv_cache_only=False so we get the FFN activations too!
    calib_manager = InfinityCalibManager(
        model=model,
        config=config,
        other_configs=unused_cfgs,
        smooth_cache={},
        save_kv_cache_only=False, 
        save_imgs=False
    )
    
    stats_log = {}
    
    # 3. Iterate Layers
    print("Collecting Activation Statistics...")
    data_iterator = calib_manager.iter_layer_activations()
    
    for block_struct, aggregated_cache, block_kwargs in tqdm(data_iterator, desc="Analyzing Blocks"):
        block_name = block_struct.name
        
        # --- A. Analyze Linear Layers (FFN Down Proj) ---
        if block_name+'.sa.to_q' in aggregated_cache:
            # aggregated_cache['ffn_fc2'] is an IOTensorsCache. We extract the tensor.
            # It stores a list of tensors (one per batch usually). We cat them.
            tensor_list = aggregated_cache[block_name+'.sa.to_q'].inputs.tensors[0].data # Access internal list
            if tensor_list:
                # tensor_list is likely [Batch1_Tensor, Batch2_Tensor...]
                # Each tensor is [1, Seq, Dim]
                full_tensor = torch.cat(tensor_list, dim=1).to("cuda") # [TotalBatch, Seq, Dim]
                analyze_layer(stats_log, f"{block_name}.to_qkv", full_tensor, layer_type="linear")
        if block_name+'.ffn.fc1' in aggregated_cache:
            # aggregated_cache['ffn_fc2'] is an IOTensorsCache. We extract the tensor.
            # It stores a list of tensors (one per batch usually). We cat them.
            tensor_list = aggregated_cache[block_name+'.ffn.fc1'].inputs.tensors[0].data # Access internal list
            if tensor_list:
                # tensor_list is likely [Batch1_Tensor, Batch2_Tensor...]
                # Each tensor is [1, Seq, Dim]
                full_tensor = torch.cat(tensor_list, dim=1).to("cuda") # [TotalBatch, Seq, Dim]
                analyze_layer(stats_log, f"{block_name}.ffn_up", full_tensor, layer_type="linear")
        if block_name+'.ffn.fc2' in aggregated_cache:
            # aggregated_cache['ffn_fc2'] is an IOTensorsCache. We extract the tensor.
            # It stores a list of tensors (one per batch usually). We cat them.
            tensor_list = aggregated_cache[block_name+'.ffn.fc2'].inputs.tensors[0].data # Access internal list
            if tensor_list:
                # tensor_list is likely [Batch1_Tensor, Batch2_Tensor...]
                # Each tensor is [1, Seq, Dim]
                full_tensor = torch.cat(tensor_list, dim=1).to("cuda") # [TotalBatch, Seq, Dim]
                analyze_layer(stats_log, f"{block_name}.ffn_down", full_tensor, layer_type="linear")
        if block_name+'.sa.proj' in aggregated_cache:
            # aggregated_cache['ffn_fc2'] is an IOTensorsCache. We extract the tensor.
            # It stores a list of tensors (one per batch usually). We cat them.
            tensor_list = aggregated_cache[block_name+'.sa.proj'].inputs.tensors[0].data # Access internal list
            if tensor_list:
                # tensor_list is likely [Batch1_Tensor, Batch2_Tensor...]
                # Each tensor is [1, Seq, Dim]
                full_tensor = torch.cat(tensor_list, dim=1).to("cuda") # [TotalBatch, Seq, Dim]
                analyze_layer(stats_log, f"{block_name}.out_proj", full_tensor, layer_type="linear")
        
        # --- B. Analyze KV Cache (Self Attention) ---
        # Accessed via block_kwargs passed from the StatefulInfinity model
        iter_id = 12
        while iter_id < len(block_kwargs):
            sa_cache = block_kwargs[iter_id].get('sa_kv_cache', {}).get('sa', {})
            k_cache = sa_cache.get('k')
            v_cache = sa_cache.get('v')
            
            if k_cache is not None:
                # Shape is usually [Batch, Heads, Seq, Dim] or [Batch, Seq, Heads, Dim]
                # We flatten to [TotalTokens, TotalChannels] to analyze distribution
                # Assume standard [B, H, S, D]. Flatten to [B*S, H*D]
                
                # Check dim to be safe
                if k_cache.dim() == 4: 
                    k_flat = k_cache.permute(0, 2, 1, 3).reshape(-1, k_cache.shape[1] * k_cache.shape[3])
                    v_flat = v_cache.permute(0, 2, 1, 3).reshape(-1, v_cache.shape[1] * v_cache.shape[3])
                    
                    analyze_layer(stats_log, f"{block_name}.attn_k", k_flat, layer_type="kv_cache")
                    analyze_layer(stats_log, f"{block_name}.attn_v", v_flat, layer_type="kv_cache")
            
            iter_id += GENERATIVE_STEPS

        del block_struct
        del aggregated_cache
        del block_kwargs
        gc.collect()

        # 4. Aggregate & Print Report
        print("\n" + "="*80)
        print(f"{'Layer Type':<20} | {'Max/Med':<10} | {'Kurtosis':<10} | {'CV_Chan':<10} | {'CV_Tok':<10} | {'Skew':<10} | {'Max Skew':<10}")
        print("="*80)
        
        # Aggregating FFN stats
        to_qkv_ratios = [x['max_median_ratio'] for k,v in stats_log.items() if 'to_qkv' in k for x in v]
        to_qkv_kurt = [x['kurtosis'] for k,v in stats_log.items() if 'to_qkv' in k for x in v]
        ffn1_ratios = [x['max_median_ratio'] for k,v in stats_log.items() if 'ffn_up' in k for x in v]
        ffn1_kurt = [x['kurtosis'] for k,v in stats_log.items() if 'ffn_up' in k for x in v]
        ffn2_ratios = [x['max_median_ratio'] for k,v in stats_log.items() if 'ffn_down' in k for x in v]
        ffn2_kurt = [x['kurtosis'] for k,v in stats_log.items() if 'ffn_dow' in k for x in v]
        out_proj_ratios = [x['max_median_ratio'] for k,v in stats_log.items() if 'out_proj' in k for x in v]
        out_proj_kurt = [x['kurtosis'] for k,v in stats_log.items() if 'out_proj' in k for x in v]
        
        # Aggregating KV stats 
        k_cv_c = [x['cv_channel'] for k,v in stats_log.items() if 'attn_k' in k for x in v]
        k_cv_t = [x['cv_token'] for k,v in stats_log.items() if 'attn_k' in k for x in v]
        v_cv_c = [x['cv_channel'] for k,v in stats_log.items() if 'attn_v' in k for x in v]
        v_cv_t = [x['cv_token'] for k,v in stats_log.items() if 'attn_v' in k for x in v]
        skew_k = [x['skewness'] for k,v in stats_log.items() if 'attn_k' in k for x in v]
        skew_v = [x['skewness'] for k,v in stats_log.items() if 'attn_v' in k for x in v]
        max_skew_k = [x['max_skew'] for k,v in stats_log.items() if 'attn_k' in k for x in v]   
        max_skew_v = [x['max_skew'] for k,v in stats_log.items() if 'attn_v' in k for x in v]
        
        print(f"{'QKV Projections':<20} | {np.mean(to_qkv_ratios):<10.1f} | {np.mean(to_qkv_kurt):<10.1f} | {'-':<10} |")
        print(f"{'FFN Up Proj':<20} | {np.mean(ffn1_ratios):<10.1f} | {np.mean(ffn1_kurt):<10.1f} | {'-':<10} |")
        print(f"{'FFN Down Proj':<20} | {np.mean(ffn2_ratios):<10.1f} | {np.mean(ffn2_kurt):<10.1f} | {'-':<10} |")
        print(f"{'Output Proj':<20} | {np.mean(out_proj_ratios):<10.1f} | {np.mean(out_proj_kurt):<10.1f} | {'-':<10} |")
        
        if k_cv_c:
            dom = "Channel" if np.mean(k_cv_c) > np.mean(k_cv_t) else "Token"
            print(f"{'Attn Key Cache':<20} | {'-':<10} | {'-':<10} | {np.mean(k_cv_c):<10.2f} | {np.mean(k_cv_t):<10.2f} | {np.mean(skew_k):<10.2f} |{np.mean(max_skew_k):<10.2f}")

        if v_cv_c:
            dom = "Channel" if np.mean(v_cv_c) > np.mean(v_cv_t) else "Token"
            print(f"{'Attn Value Cache':<20} | {'-':<10} | {'-':<10} | {np.mean(v_cv_c):<10.2f} | {np.mean(v_cv_t):<10.2f} | {np.mean(skew_v):<10.2f}|{np.mean(max_skew_v):<10.2f}")
        print("="*80)

    # Save raw json
    with open(f"infinity_stats_{nname}.json", "w") as f:
        json.dump(stats_log, f, indent=2)

if __name__ == "__main__":
    config, _, unused_cfgs, _, _ = DiffusionPtqRunConfig.get_parser().parse_known_args()
    main(config, unused_cfgs)