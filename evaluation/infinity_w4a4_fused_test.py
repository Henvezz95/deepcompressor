import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import argparse 
import cv2
import copy
import numpy as np


# --- Configure Python Path to Find Modules ---
#sys.path.append('./') 
sys.path.append('./Infinity_rep') 


# We need to import the loader functions
from tools.run_infinity import load_visual_tokenizer, load_transformer, load_tokenizer, gen_one_img, h_div_w_templates, dynamic_resolution_h_w
from evaluation.quantized_layers import SVDQuantLinear, swap_mlp_for_svdquant_fused, load_svdquant_weights
from evaluation.quantized_layers import swap_linear_for_svdquant, load_svdquant_artifacts, SVDQuantFusedMLP

# --- Main Test Execution ---
def main():
    print("--- Loading a real Infinity model from checkpoint ---")
    
    
    args = argparse.Namespace(
        pn='1M', model_path='/workspace/Infinity/weights/infinity_2b_reg.pth',
        vae_path='/workspace/Infinity/weights/infinity_vae_d32reg.pth',
        text_encoder_ckpt='/workspace/Infinity/weights/flan-t5-xl',
        model_type='infinity_2b', vae_type=32, text_channels=2048,
        add_lvl_embeding_only_first_block=1, use_bit_label=1,
        rope2d_each_sa_layer=1, rope2d_normalized_by_hw=2, apply_spatial_patchify=0,
        cfg_insertion_layer=0, use_scale_schedule_embedding=0, sampling_per_bits=1,
        h_div_w_template=1.000, use_flex_attn=0, cache_dir='/dev/shm',
        checkpoint_type='torch', seed=0, bf16=1, save_file='tmp.jpg',
        enable_model_cache=0
    )
    '''


    args=argparse.Namespace(
        pn='1M', model_path='/workspace/deepcompressor/Infinity_rep/weights/infinity_8b_weights',
        vae_path='/workspace/Infinity/weights/infinity_vae_d56_f8_14_patchify.pth',
        text_encoder_ckpt='/workspace/Infinity/weights/flan-t5-xl',
        cfg_insertion_layer=0, vae_type=14, add_lvl_embeding_only_first_block=1,
        use_bit_label=1, model_type='infinity_8b', rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2, use_scale_schedule_embedding=0, sampling_per_bits=1,
        text_channels=2048, apply_spatial_patchify=1, h_div_w_template=1.000,
        use_flex_attn=0, cache_dir='/dev/shm', checkpoint_type='torch_shard',
        bf16=1, save_file='tmp.jpg'
    )'''

    
    vae = load_visual_tokenizer(args)
    model = load_transformer(vae, args).eval()

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)

    h_div_w = 1/1
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    torch.cuda.reset_peak_memory_stats()
    
    
    print(f"\n--- Benchmarking {model.__class__.__name__} ---")

    '''
    for i in range(5):
        img = gen_one_img(
            model,
            vae,
            text_tokenizer,
            text_encoder,
            'A photorealistic image of a dog working ' \
            'as a professor at MIT university giving a lesson ' \
            'on Autoregressive Models for NLP',
            g_seed=16,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=3.0,
            tau_list=0.5,
            scale_schedule=scale_schedule, 
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=True,
        )

        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_gb = peak_memory_bytes / (1024**3)
        print(f"Peak GPU Memory Usage: {peak_memory_gb:.2f} GB")'''

    #cv2.imwrite('img.png', img.detach().cpu().numpy())

    # Create the full W4A4 (SVDQuant) version
    print("\n--- Creating full W4A4 (SVDQuant) model ---")
    #svdquant_model = copy.deepcopy(model)
    svdquant_model = model


    swap_linear_for_svdquant(svdquant_model, lora_rank=32, exclude_names=['word_embed', 'head', 
                                                                          'text_norm', 'norm0_cond', 
                                                                          'text_proj_for_sos', 'text_proj_for_ca', 
                                                                          'lvl_embed', 'shared_ada_lin', 
                                                                          'head_nm', 'ca.mat_kv', 'block_chunks.5.module.0'])
    
    replaced = swap_mlp_for_svdquant_fused(
        svdquant_model,
        lora_rank=32,          # or 0 for no lora
        use_fp4=False,         # toggle if you want FP4 path
        only_gelu=True,        # conservative: fuse GELU paths only
        group_size=64,         # matches our synthetic per-group scales
        exclude_names=["text_proj_for_ca.*"]
    )

    base_path = '../deepcompressor/runs/diffusion/int4_rank32_batch12/model/'
    arts = load_svdquant_artifacts(base_path)  # contains model.pt, scale.pt, branch.pt, smooth.pt
    report = load_svdquant_weights(svdquant_model, arts, strict=True, dry_run=False)
    # --- IMPORTANT: CLEAR MEMORY ---
    #del model
    torch.cuda.empty_cache()
    svdquant_model_to_run = svdquant_model.eval()
    # Reset stats AFTER loading the model to memory
    torch.cuda.reset_peak_memory_stats()

    for i in range(5):
        img = gen_one_img(
            svdquant_model_to_run,
            vae,
            text_tokenizer,
            text_encoder,
            'A photorealistic image of a dog working as a professor at MIT university giving ' \
            'a lesson on Autoregressive Models for NLP',
            g_seed=16,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=3.0,
            tau_list=0.5,
            scale_schedule=scale_schedule, 
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=False,
        )
        cv2.imwrite('img.png', img.detach().cpu().numpy())

    svdquant_peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"SVDQuant Peak Memory: {svdquant_peak_mem:.2f} GB")

    # --- IMPORTANT: CLEAR MEMORY ---
    del svdquant_model_to_run
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()