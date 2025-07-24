import torch
import torch.nn as nn
import os

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

# --- Main Debugging Logic ---

# 1. Load the "black box" golden model and all artifacts
print("--- Loading Golden Model and All Artifacts ---")
golden_model = torch.load('runs/diffusion/int4_rank32_batch12/model/golden_reference.pkl', weights_only=False).eval()
base_path = 'runs/diffusion/int4_rank32_batch12/model/'
weights_state_dict = torch.load(os.path.join(base_path, 'model.pt'))
branch_state_dict = torch.load(os.path.join(base_path, 'branch.pt'))
smooth_scales = torch.load(os.path.join(base_path, 'smooth.pt'))
configs, _, _, _, _ = DiffusionPtqRunConfig.get_parser().parse_known_args()
config = configs.quant


# 2. Choose a target layer to test
# Let's pick the first FFN's input layer for simplicity.
layer_name = "block_chunks.0.module.0.ffn.fc2"
target_golden_layer = golden_model.get_submodule(layer_name)


# 3. Create the "white box" reconstructed layer
print(f"--- Reconstructing a manual version of '{layer_name}' ---")
reconstructed_layer = nn.Linear(
    target_golden_layer.in_features,
    target_golden_layer.out_features,
    bias=True # Assuming bias is false, adjust if needed
).to(device=target_golden_layer.weight.device, dtype=target_golden_layer.weight.dtype)

# Load its weights from model.pt
reconstructed_layer.weight.data.copy_(weights_state_dict[f"{layer_name}.weight"])
reconstructed_layer.bias.data.copy_(weights_state_dict[f"{layer_name}.bias"])

# Attach its LowRankBranch hook from branch.pt
smoother = ActivationSmoother(smooth_scales[layer_name], channels_dim=-1)
smoother.input_packager = SimpleInputPackager()  # Use the actual class name
smoother.as_hook().register(reconstructed_layer)


branch = LowRankBranch(
    in_features=reconstructed_layer.in_features,
    out_features=reconstructed_layer.out_features,
    rank=configs.quant.wgts.low_rank.rank
).to(device=reconstructed_layer.weight.device, dtype=reconstructed_layer.weight.dtype)
branch.load_state_dict(branch_state_dict[layer_name])
branch.input_packager = SimpleInputPackager() 
branch.as_hook().register(reconstructed_layer)

# Attach the FUSED SmoothQuant hook
#quantizer = Quantizer(config.ipts, key=layer_name, channels_dim=-1)
#quantizer.smooth_scale = smooth_scales[layer_name] # Configure with the scale!
#quantizer.input_packager = SimpleInputPackager()
#quantizer.as_hook().register(reconstructed_layer)

# Create a random tensor to run a single forward pass
# NOTE: To be perfectly accurate, you should run the full `gen_one_img`
# and have the hook capture an input during that process.
# For a quick test, a random tensor is often sufficient.
sample_input = torch.randn(
    1, 15, 8192, # Assuming an input size for ffn.fc1 for example
    device=next(golden_model.parameters()).device,
    dtype=next(golden_model.parameters()).dtype
)
with torch.no_grad():
    # Pass the sample input through the golden layer to trigger the hooks
    golden_output = target_golden_layer(sample_input)

assert torch.allclose(target_golden_layer.weight.data, reconstructed_layer.weight.data, atol=1e-5)
assert torch.allclose(target_golden_layer.bias.data, reconstructed_layer.bias.data, atol=1e-5)

# 5. Pass the captured input through the reconstructed layer
print("--- Running Inference on Reconstructed Layer ---")
with torch.no_grad():
    reconstructed_output = reconstructed_layer(sample_input)

# 6. The Definitive Comparison
print("\n--- Verification ---")

are_outputs_close = torch.allclose(golden_output, reconstructed_output, atol=1e-3)
print(f"Does the output of the golden layer match the reconstructed layer? -> {are_outputs_close}")
if not are_outputs_close:
    print(f"Max difference: {(golden_output - reconstructed_output).abs().max().item()}")
    print(f"Mean absolute difference: {(golden_output - reconstructed_output).abs().mean().item()}")

