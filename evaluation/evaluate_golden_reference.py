import os
import cv2
import torch
from torch import nn
import argparse
import numpy as np
from collections import OrderedDict
import sys
sys.path.append('../')
import omniconfig

# Assuming all your custom and library imports are correctly set up
# (Imports from previous files are included here for completeness)
from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct, patchModel, DiffusionAttentionStruct
from deepcompressor.app.diffusion.quant.weight import calibrate_diffusion_block_low_rank_branch
from deepcompressor.calib.smooth import ActivationSmoother
from deepcompressor.quantizer import Quantizer
from deepcompressor.utils.hooks import SimpleInputPackager
from deepcompressor.app.diffusion.nn.struct import DiTStruct
from deepcompressor.app.diffusion.config import DiffusionQuantConfig, DiffusionPtqRunConfig


from Infinity_rep.infinity.models.infinity import Infinity
from Infinity_rep.tools.run_infinity import load_visual_tokenizer, load_tokenizer, gen_one_img, load_transformer
from Infinity_rep.infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from deepcompressor.nn.patch.lowrank import LowRankBranch # Make sure to import this
# --- WARNING: This script is for debugging only. Loading pickled models is not recommended for production. ---

from evaluation.build_functions import assemble_model, attach_kv_qparams


# Setup arguments (keep your existing args setup)
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

configs, _, unused_cfgs, unused_args, unknown_args = DiffusionPtqRunConfig.get_parser().parse_known_args()

#  Extract the config objects as before
config = configs.quant

vae = load_visual_tokenizer(args)
model = load_transformer(vae, args)

text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)

h_div_w = 1/1
h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

# --- Load the Golden Reference Model ---
print("--- Loading the 'golden reference' model object ---")
try:
    # Load the single file containing the model and all its hooks
    quantized_model = torch.load('runs/diffusion/int4_rank32_batch12/model/golden_reference.pkl', weights_only=False)
    quantized_model.eval()
    print("✅ 'golden_reference_model.pkl' loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load the pickled model. This can happen if the code has changed since saving. Error: {e}")
    exit()

model_struct = DiTStruct.construct(quantized_model)

#attach_kv_qparams(quantized_model, os.path.join('runs/', "kv_scales", "kv_quant_calib.pt"))

'''
for module_key, module_name, module, _, _ in model_struct.named_key_modules():
    # Check if this layer should have its activations quantized
    if config.ipts.is_enabled_for(module_key):
        print(f"Attaching ACTIVE A4 Quantizer hook to: {module_name}")
        quantizer = Quantizer(config.ipts, key=module_name, channels_dim=-1)
        
        # The crucial fix:
        quantizer.input_packager = SimpleInputPackager()
        quantizer.as_hook().register(module)
'''

print("✅ Activation hooks are now truly active.")

prompt = 'A photo of a happy dog'
save_name = "_".join(prompt.split(' '))

'''
# Reference Image
img = gen_one_img(
    model,
    vae,
    text_tokenizer,
    text_encoder,
    prompt,
    g_seed=16,
    gt_leak=0,
    gt_ls_Bl=None,
    cfg_list=[3.0] * len(scale_schedule),
    tau_list=[0.5] * len(scale_schedule),
    scale_schedule=scale_schedule,
    cfg_insertion_layer=[args.cfg_insertion_layer],
    vae_type=args.vae_type,
    sampling_per_bits=args.sampling_per_bits,
    enable_positive_prompt=False,
)
cv2.imwrite(f'{save_name}_FP16.jpg', img.detach().cpu().numpy())
print(f"Generated test image: {save_name}_FP16.jpg")
'''
# Now the quantized_model is fully assembled and ready for testing
img = gen_one_img(
    quantized_model,
    vae,
    text_tokenizer,
    text_encoder,
    prompt,
    g_seed=16,
    gt_leak=0,
    gt_ls_Bl=None,
    cfg_list=[3.0] * len(scale_schedule),
    tau_list=[0.5] * len(scale_schedule),
    scale_schedule=scale_schedule,
    cfg_insertion_layer=[args.cfg_insertion_layer],
    vae_type=args.vae_type,
    sampling_per_bits=args.sampling_per_bits,
    enable_positive_prompt=False,
)
cv2.imwrite(f'{save_name}_W4A16.jpg', img.detach().cpu().numpy())
print(f"Generated test image: {save_name}_W4A16.jpg")

# Assumiamo che 'golden_model' sia il modello in memoria alla fine di ptq.py
# e 'test_model' sia quello che carichi da disco nel tuo script di test.

# --- 1. Estrai i pesi dal GOLDEN REFERENCE (in-memory) ---
# Trova l'hook specifico. L'accesso potrebbe variare leggermente.
golden_sa_module = quantized_model.block_chunks[0].module[0].sa
golden_q_proj_weight = golden_sa_module.to_q.weight.data

# Gli hook sono memorizzati in un dizionario OrderedDict chiamato _forward_hooks
golden_hook_dict = golden_sa_module.to_q._forward_hooks
# Assumiamo che l'hook LRA sia il primo (o l'unico)
golden_lra_hook = list(golden_hook_dict.values())[0]
golden_a = golden_lra_hook.branch.a.weight.data
golden_b = golden_lra_hook.branch.b.weight.data


# --- 2. Estrai i pesi dai FILE CARICATI (.pt) ---
# Carica i dizionari come fai nel tuo script di test
base_path = 'runs/diffusion/int4_rank32_batch12/model/'
branch_state_dict = torch.load(os.path.join(base_path, 'branch.pt'), map_location='cuda:0')
weights_state_dict = torch.load(os.path.join(base_path, 'model.pt'), weights_only=True, map_location='cuda:0')

# I nomi delle chiavi devono corrispondere a come sono stati salvati
q_proj_name = "block_chunks.0.module.0.sa.to_q" # Adatta questo nome se necessario
loaded_q_proj_weight = weights_state_dict[f"{q_proj_name}.weight"]

# Il branch per q,k,v è condiviso e salvato sotto il nome di 'to_q'
branch_key = q_proj_name 
loaded_a = branch_state_dict[branch_key]['a.weight']
loaded_b = branch_state_dict[branch_key]['b.weight']


# --- 3. Confronto Definitivo ---
print("--- Confronto Definitivo dei Tensori ---")
print(f"Primary Weight ('to_q') corrisponde? -> {torch.allclose(golden_q_proj_weight, loaded_q_proj_weight)}")
print(f"Matrice 'a' corrisponde?             -> {torch.allclose(golden_a, loaded_a)}")
print(f"Matrice 'b' (primi 2048 canali) corrisponde? -> {torch.allclose(golden_b, loaded_b[:2048, :])}")

# 1. Extract GOLDEN REFERENCE weights and hooks for the '.to_k' layer
print("--- Accessing Golden Reference In-Memory Objects for '.to_k'---")
golden_ca_module = quantized_model.block_chunks[4].module[2].ca
golden_k_proj_module = golden_ca_module.to_k

# Get the primary weight from the .to_k layer itself
golden_k_proj_weight = golden_k_proj_module.weight.data

# Get the hook that is attached specifically to the .to_k layer
golden_k_hook = list(golden_k_proj_module._forward_hooks.values())[0]
golden_a_for_k = golden_k_hook.branch.a.weight.data
golden_b_for_k = golden_k_hook.branch.b.weight.data

# --- 2. Extract weights for '.to_k' from LOADED FILES ---
print("--- Loading Artifacts from .pt Files ---")
base_path = 'runs/diffusion/int4_rank32_batch12/model/'
branch_state_dict = torch.load(os.path.join(base_path, 'branch.pt'), map_location='cuda:0')
weights_state_dict = torch.load(os.path.join(base_path, 'model.pt'), weights_only=True, map_location='cuda:0')

# Get the primary weight for '.to_k' from model.pt
k_proj_name = "block_chunks.4.module.2.ca.to_k"
loaded_k_proj_weight = weights_state_dict[f"{k_proj_name}.weight"]

# Get the FUSED branch from branch.pt (it's under the 'to_q' key)
fused_branch_key = "block_chunks.4.module.2.ca.to_q"
loaded_fused_branch = branch_state_dict[fused_branch_key]

# The 'a' matrix is the shared one from the fused branch
loaded_a_shared = loaded_fused_branch['a.weight']
# The 'b' matrix is the SECOND slice (for K) of the fused 'b' matrix
loaded_b_for_k = loaded_fused_branch['b.weight'][2048:4096, :]


# --- 3. Definitive Comparison for '.to_k' ---
print("\n--- Definitive Comparison for Fused Cross-Attention '.to_k' ---")
print(f"Primary Weight ('.to_k') corresponds? -> {torch.allclose(golden_k_proj_weight, loaded_k_proj_weight)}")
print(f"Matrix 'a' (Shared) corresponds?      -> {torch.allclose(golden_a_for_k, loaded_a_shared)}")
print(f"Matrix 'b' (K-slice) corresponds?     -> {torch.allclose(golden_b_for_k, loaded_b_for_k)}")