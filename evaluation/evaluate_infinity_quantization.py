import os
import cv2
import torch
from torch import nn
import argparse
import numpy as np
from collections import OrderedDict
import omniconfig
import gc

# Assuming all your custom and library imports are correctly set up
# (Imports from previous files are included here for completeness)
from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct, patchModel, DiffusionAttentionStruct
from deepcompressor.app.diffusion.quant.weight import calibrate_diffusion_block_low_rank_branch
from deepcompressor.calib.smooth import ActivationSmoother
from deepcompressor.quantizer import Quantizer
from deepcompressor.utils.hooks import SimpleInputPackager
from deepcompressor.app.diffusion.config import DiffusionQuantConfig, DiffusionPtqRunConfig


from Infinity_rep.infinity.models.infinity import Infinity
from Infinity_rep.tools.run_infinity import load_visual_tokenizer, load_tokenizer, gen_one_img, load_transformer
from Infinity_rep.infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from deepcompressor.nn.patch.lowrank import LowRankBranch # Make sure to import this

from evaluation.build_functions import assemble_model


# --- Your Original Setup Code ---
args = argparse.Namespace(
    pn='1M', model_path='./Infinity_rep/weights/infinity_2b_reg.pth',
    vae_path='./Infinity_rep/weights/infinity_vae_d32reg.pth',
    text_encoder_ckpt='./Infinity_rep/weights/flan-t5-xl',
    model_type='infinity_2b', vae_type=32, text_channels=2048,
    add_lvl_embeding_only_first_block=1, use_bit_label=1,
    rope2d_each_sa_layer=1, rope2d_normalized_by_hw=2, apply_spatial_patchify=0,
    cfg_insertion_layer=0, use_scale_schedule_embedding=0, sampling_per_bits=1,
    h_div_w_template=1.000, use_flex_attn=0, cache_dir='/dev/shm',
    checkpoint_type='torch', seed=0, bf16=0, save_file='tmp.jpg',
    enable_model_cache=0
)

# You will need your quantization config to initialize the activation quantizers
# This is a placeholder for your actual config loading
ptq_config, _, unused_cfgs, unused_args, unknown_args = DiffusionPtqRunConfig.get_parser().parse_known_args()
ptq_config.output.lock()

vae = load_visual_tokenizer(args)
# NOTE: The model should be loaded in float32 for this testing phase
model = load_transformer(vae, args)

text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)

h_div_w = 1/1
h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

print("--- Patching attention layers to be compatible ---")
quantized_model = patchModel(model)
print("Patching complete.\n")

# --- START OF NEW CODE: FINAL MODEL ASSEMBLY ---

# 1. Load all three necessary files
print("--- Loading weights, smoothing scales, and low-rank branches ---")
base_path = 'runs/diffusion/int4_rank32_batch12/model/' 
weights = torch.load(os.path.join(base_path, 'model.pt'))
smooth_scales = torch.load(os.path.join(base_path, 'smooth.pt'))
branch_state_dict = torch.load(os.path.join(base_path, 'branch.pt'))

# Load the final weights into the model
quantized_model.load_state_dict(weights)

# Create a struct to easily iterate through the model
model_struct = InfinityStruct.construct(quantized_model)
for name, module in quantized_model.named_modules():
    module.name = name

generation_args = { 'cfg_list': [3.0]*13, 'tau_list': [0.5]*13, 'g_seed': 16,
                    'gt_leak': 0, 'gt_ls_Bl': None, 'scale_schedule': scale_schedule,
                    'cfg_insertion_layer': [args.cfg_insertion_layer], 'vae_type': args.vae_type,
                    'sampling_per_bits': args.sampling_per_bits, 'enable_positive_prompt': True }


model_struct = assemble_model(model_struct, 
                              ptq_config, 
                              branch_state_dict, 
                              smooth_scales, 
                              weights, 
                              quantize_activations = True,
                              skip_ca_kv_act = False)

del weights
del branch_state_dict
del smooth_scales
gc.collect()

# Now the quantized_model is fully assembled and ready for testing
img = gen_one_img(
    quantized_model,
    vae,
    text_tokenizer,
    text_encoder,
    'A photo of a happy dog',
    g_seed=16,
    gt_leak=0,
    gt_ls_Bl=None,
    cfg_list=[3.0] * len(scale_schedule),
    tau_list=[0.5] * len(scale_schedule),
    scale_schedule=scale_schedule,
    cfg_insertion_layer=[args.cfg_insertion_layer],
    vae_type=args.vae_type,
    sampling_per_bits=args.sampling_per_bits,
    enable_positive_prompt=True,
)

file_name = 'img_patched_quantized_w4a4.jpg'
cv2.imwrite(file_name, img.detach().cpu().numpy())
print(f"Generated test image: {file_name}")
