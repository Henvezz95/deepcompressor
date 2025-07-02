import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
from dataclasses import dataclass, field
from functools import partial
import math
import sys
import argparse 
import cv2
import numpy as np

from flash_attn import flash_attn_varlen_kvpacked_func

# --- Configure Python Path to Find Modules ---
sys.path.append('./') 
sys.path.append('./Infinity_rep') 


# We need to import the loader functions
from tools.run_infinity import load_visual_tokenizer, load_transformer, load_tokenizer, gen_one_img, h_div_w_templates, dynamic_resolution_h_w
from struct_infinity import patchModel
from torch.quantization import quantize_dynamic
from torch.ao.quantization import QConfigMapping, get_default_qconfig

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
    
    vae = load_visual_tokenizer(args)
    model = load_transformer(vae, args)

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)

    h_div_w = 1/1
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    prompt = 'A photo of a happy dog'
    '''
    img = gen_one_img(
        model,
        vae,
        text_tokenizer,
        text_encoder,
        prompt,
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
    cv2.imwrite('img.png', img.detach().cpu().numpy())
    
    print("Full Infinity model loaded successfully.\n")
    '''
    torch.backends.quantized.engine = 'qnnpack'

    # 2. Load your original, un-patched Infinity model
    # model = load_original_infinity_model(...)
    patched_model = patchModel(model)
    patched_model.eval().cpu()

    # 2. Set the quantization engine
    torch.backends.quantized.engine = 'qnnpack'

    # 3. Create a configuration that applies to all Linear layers by default
    qconfig = get_default_qconfig('qnnpack')
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    problematic_module_name = 'text_proj_for_sos.ca.mat_kv' # <--- ADJUST THIS NAME AS NEEDED
    qconfig_mapping = qconfig_mapping.set_module_name(problematic_module_name, None)


    # 5. Prepare the model for quantization using this specific mapping
    #    (This step is needed for this more advanced configuration)
    from torch.ao.quantization import prepare
    prepared_model = prepare(patched_model, qconfig_mapping)

    # 6. Convert the model. This is the equivalent of the `quantize_dynamic` call
    from torch.ao.quantization import convert
    model_int8 = convert(prepared_model)

    print("\n--- Quantized INT8 Model ---")
    img = gen_one_img(
        model_int8,
        vae.cpu(),
        text_tokenizer,
        text_encoder.cpu(),
        prompt,
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
    cv2.imwrite('img_quantized_8bit.png', img.detach().cpu().numpy())
    

if __name__ == "__main__":
    main()
