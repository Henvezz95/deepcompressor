import os
import cv2
import torch
import argparse
import numpy as np
import gc
from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct, patchModel
from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from Infinity_rep.tools.run_infinity import load_visual_tokenizer, load_tokenizer, gen_one_img, load_transformer
from Infinity_rep.infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

from build_functions import assemble_model

# Setup arguments
args = argparse.Namespace(
    pn='1M', model_path='/workspace/Infinity/weights/infinity_2b_reg.pth',
    vae_path='/workspace/Infinity/weights/infinity_vae_d32reg.pth',
    text_encoder_ckpt='/workspace/Infinity/weights/flan-t5-xl',
    model_type='infinity_2b', vae_type=32, text_channels=2048,
    add_lvl_embeding_only_first_block=1, use_bit_label=1,
    rope2d_each_sa_layer=1, rope2d_normalized_by_hw=2, apply_spatial_patchify=0,
    cfg_insertion_layer=0, use_scale_schedule_embedding=0, sampling_per_bits=1,
    h_div_w_template=1.000, use_flex_attn=0, cache_dir='/dev/shm',
    checkpoint_type='torch', seed=0, bf16=0, save_file='tmp.jpg',
    enable_model_cache=0
)

# Load configuration from YAML file correctly
parser = DiffusionPtqRunConfig.get_parser()
config_path = 'examples/diffusion/configs/svdquant/int4.yaml'
configs, _, unused_cfgs, unused_args, unknown_args = DiffusionPtqRunConfig.get_parser().parse_known_args()

#  Extract the config objects as before
config = configs.quant

# Load base models
vae = load_visual_tokenizer(args)
model = load_transformer(vae, args)
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)

# Setup generation schedule
h_div_w = 1/1
h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

print("--- Patching attention layers to be compatible ---")
quantized_model = patchModel(model)
print("Patching complete.\n")


# 1. Load the saved artifacts for inference
print("--- Loading inference artifacts (model.pt, branch.pt) ---")
dtype = torch.bfloat16
base_path = 'runs/diffusion/int4_rank32_batch12/model/' 
weights = torch.load(os.path.join(base_path, 'model.pt'))
smooth_scales = torch.load(os.path.join(base_path, 'smooth.pt'))
branch_state_dict = torch.load(os.path.join(base_path, 'branch.pt'))

# 2. Crete the model Structure
model_struct = InfinityStruct.construct(quantized_model)
for name, module in quantized_model.named_modules():
    module.name = name

model_struct = assemble_model(model_struct, configs, branch_state_dict, smooth_scales, weights, True)

del weights
del branch_state_dict
del smooth_scales
gc.collect()
print("--- Final model assembly complete. Running inference. ---\n")

# --- Run Inference Test ---
img = gen_one_img(
    quantized_model,
    vae,
    text_tokenizer,
    text_encoder,
    'A doggo',
    g_seed=16, # Use a fixed seed for reproducibility
    cfg_list=[3.0] * len(scale_schedule),
    tau_list=[0.5] * len(scale_schedule),
    scale_schedule=scale_schedule,
    cfg_insertion_layer=[args.cfg_insertion_layer],
    vae_type=args.vae_type,
    sampling_per_bits=args.sampling_per_bits,
    enable_positive_prompt=False,
)
cv2.imwrite('img_patched_quantized.png', img.detach().cpu().numpy())
print("Generated test image: img_patched_quantized.png")
