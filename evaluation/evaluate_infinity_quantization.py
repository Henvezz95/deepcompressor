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
from deepcompressor.app.diffusion.config import DiffusionQuantConfig, DiffusionPtqRunConfig


from Infinity_rep.infinity.models.infinity import Infinity
from Infinity_rep.tools.run_infinity import load_visual_tokenizer, load_tokenizer, gen_one_img, load_transformer
from Infinity_rep.infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from deepcompressor.nn.patch.lowrank import LowRankBranch # Make sure to import this


def attach_low_rank_branches(
    model_struct: InfinityStruct,
    config: DiffusionQuantConfig, # Config is needed for rank
    branch_state_dict: dict[str, dict[str, torch.Tensor]]
):
    """
    Correctly attaches pre-computed low-rank branches to a model, handling the
    specific shared-input groupings of the Infinity architecture.
    """
    print("--- Attaching pre-computed low-rank branches with grouping awareness ---")
    
    processed_modules = set()
    
    for block_struct in model_struct.iter_transformer_block_structs():
        # --- 1. Handle Attention Blocks (Self and Cross) ---
        for attn in block_struct.attn_structs:
            if not isinstance(attn, DiffusionAttentionStruct):
                continue
            print(attn.name)
            # --- Case A: Self-Attention (Q, K, and V are grouped) ---
            if '.sa' in attn.name:
                group_modules = attn.qkv_proj
                group_names = attn.qkv_proj_names
                first_module_name = attn.q_proj_name

                if first_module_name in branch_state_dict and first_module_name not in processed_modules:
                    print(f"Attaching shared QKV branch for group starting with {first_module_name}")
                    
                    shared_state_dict = branch_state_dict[first_module_name]
                    rank = shared_state_dict['a.weight'].shape[0]
                    shared_branch = LowRankBranch(
                        in_features=shared_state_dict['a.weight'].shape[1],
                        out_features=shared_state_dict['b.weight'].shape[0],
                        rank=rank
                    )
                    shared_branch.load_state_dict(shared_state_dict)

                    output_channel_offset = 0
                    # **CORRECTION**: Iterate over modules AND names together using zip
                    for module, module_name in zip(group_modules, group_names):
                        small_branch = LowRankBranch(module.in_features, module.out_features, rank)
                        small_branch.to(device=module.weight.device, dtype=module.weight.dtype)
                        small_branch.a.weight.data.copy_(shared_branch.a.weight)
                        b_weight_slice = shared_branch.b.weight[output_channel_offset : output_channel_offset + module.out_features]
                        small_branch.b.weight.data.copy_(b_weight_slice)
                        small_branch.as_hook().register(module)
                        processed_modules.add(module_name) # Now using the correct name string
                        output_channel_offset += module.out_features

            # --- Case B: Cross-Attention ---
            elif '.ca' in attn.name:
                # Handle standalone Q
                q_name = attn.q_proj_name
                if q_name in branch_state_dict and q_name not in processed_modules:
                    print(f"Attaching standalone branch for: {q_name}")
                    state_dict = branch_state_dict[q_name]
                    rank = config.wgts.low_rank.rank
                    branch = LowRankBranch(attn.q_proj.in_features, attn.q_proj.out_features, rank)
                    branch.load_state_dict(state_dict)
                    branch.to(device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)
                    branch.as_hook().register(attn.q_proj)
                    processed_modules.add(q_name)

                # Handle grouped K, V
                kv_group_modules = [attn.k_proj, attn.v_proj]
                kv_group_names = [attn.k_proj_name, attn.v_proj_name]
                first_module_name = attn.k_proj_name

                if first_module_name in branch_state_dict and first_module_name not in processed_modules:
                    print(f"Attaching shared KV branch for group: {kv_group_names}")
                    shared_state_dict = branch_state_dict[first_module_name]
                    rank = shared_state_dict['a.weight'].shape[0]
                    shared_branch = LowRankBranch(
                        in_features=shared_state_dict['a.weight'].shape[1],
                        out_features=shared_state_dict['b.weight'].shape[0],
                        rank=rank
                    )
                    shared_branch.load_state_dict(shared_state_dict)

                    output_channel_offset = 0
                    # **CORRECTION**: Iterate over modules AND names together using zip
                    for module, module_name in zip(kv_group_modules, kv_group_names):
                        small_branch = LowRankBranch(module.in_features, module.out_features, rank)
                        small_branch.to(device=module.weight.device, dtype=module.weight.dtype)
                        small_branch.a.weight.data.copy_(shared_branch.a.weight)
                        b_weight_slice = shared_branch.b.weight[output_channel_offset : output_channel_offset + module.out_features]
                        small_branch.b.weight.data.copy_(b_weight_slice)
                        small_branch.as_hook().register(module)
                        processed_modules.add(module_name) # Now using the correct name string
                        output_channel_offset += module.out_features
        
        # --- 2. Handle Standalone Layers (FFN and Output Projections) ---
        for module_key, module_name, module, parent, field_name in block_struct.named_key_modules():
            if module_name in branch_state_dict and module_name not in processed_modules:
                if isinstance(module, torch.nn.Linear):
                    print(f"Attaching standalone branch for: {module_name}")
                    state_dict = branch_state_dict[module_name]
                    rank = state_dict['a.weight'].shape[0]
                    branch = LowRankBranch(module.in_features, module.out_features, rank)
                    branch.load_state_dict(state_dict)
                    branch.to(device=module.weight.device, dtype=module.weight.dtype)
                    branch.as_hook().register(module)
                    processed_modules.add(module_name)

    print("--- Low-rank branches attached successfully. ---")

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
configs, _, unused_cfgs, unused_args, unknown_args = DiffusionPtqRunConfig.get_parser().parse_known_args()

#  Extract the config objects as before
config = configs.quant

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
base_path = '/runs/diffusion/infinity_2b/infinity_2b/w.4-x.4-y.16/w.sint4-x.sint4.u-y.bf16/w.v64.bf16-x.v64.bf16-y.tnsr.bf16/smooth.proj-w.static.lowrank/shift-skip.x.[[w]+tan+tn].w.[e+tpo]-low.r32.i100.e.skip.[r+s+tan+tn+tpi]-smth.proj.GridSearch.bn2.[AbsMax].lr.skip.[r+s+tan+tn+tpi]-qdiff.128-t13.g3.0-s5000.RUNNING/run-250707.155345.RUNNING/model/'
#smoothing_scales = torch.load(os.path.join(base_path, 'smooth.pt'))
smoothing_scales = torch.load('smooth.pt')
branch_state_dict = torch.load(os.path.join(base_path, 'branch.pt'))
weights = torch.load(os.path.join(base_path, 'model.pt'), weights_only=True)

# Load the final weights into the model
quantized_model.load_state_dict(weights)

# Create a struct to easily iterate through the model
from deepcompressor.app.diffusion.nn.struct import DiTStruct
model_struct = DiTStruct.construct(quantized_model)

# Attach Low-Rank hooks
attach_low_rank_branches(model_struct, config, branch_state_dict)

print("--- Re-attaching all runtime hooks ---")
# Iterate through each block to attach the necessary hooks
for block_struct in model_struct.iter_transformer_block_structs():

    # 3. Attach ActivationSmoother Hooks
    # We iterate through the attention and ffn modules to attach smoothing hooks where needed.
    for attn in block_struct.attn_structs:
        # Attach hooks to all non-fusible projection layers
        for proj_name in [attn.q_proj_name, attn.k_proj_name, attn.v_proj_name, attn.o_proj_name]:
            if proj_name in smoothing_scales:
                smoother = ActivationSmoother(smoothing_scales[proj_name], channels_dim=-1)
                smoother.input_packager = SimpleInputPackager()
                module_to_hook = quantized_model.get_submodule(proj_name)
                smoother.as_hook().register(module_to_hook)

    # FFN fc1 needs a smoothing hook
    if block_struct.ffn_struct:
        ffn1_name = block_struct.ffn_struct.up_proj_name
        if ffn1_name in smoothing_scales:
            smoother = ActivationSmoother(smoothing_scales[ffn1_name], channels_dim=-1)
            smoother.input_packager = SimpleInputPackager()
            module_to_hook = quantized_model.get_submodule(ffn1_name)
            smoother.as_hook().register(module_to_hook)

    # 4. Attach Activation Quantizer Hooks
    # Finally, add the hooks that perform the dynamic activation quantization.
    for _, module_name, module, _, _ in block_struct.named_key_modules():
        if isinstance(module, torch.nn.Linear):
            # Create a quantizer based on the 'ipts' config from your YAML file
            input_quantizer = Quantizer(config.ipts, key=module_name, channels_dim=-1)
            input_quantizer.as_hook().register(module)

print("--- Final model assembly complete. Running inference. ---\n")
# --- END OF NEW CODE ---


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
cv2.imwrite('img_patched_quantized.jpg', img.detach().cpu().numpy())
print("Generated test image: img_patched_quantized.jpg")
