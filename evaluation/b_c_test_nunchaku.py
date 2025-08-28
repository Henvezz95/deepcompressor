import torch
import torch.nn as nn
import os

import os
import cv2
import torch
from torch import nn
import argparse
import torch.nn.functional as F
import math
import numpy as np
from collections import OrderedDict
import sys
sys.path.append('../')
import omniconfig

from deepcompressor.backend.nunchaku.convert import convert_to_nunchaku_w4x4y16_linear_state_dict

# Assuming all your custom and library imports are correctly set up
# (Imports from previous files are included here for completeness)
from deepcompressor.calib.smooth import ActivationSmoother
from deepcompressor.quantizer import Quantizer
from deepcompressor.utils.hooks import SimpleInputPackager
from deepcompressor.app.diffusion.nn.struct import DiTStruct
from deepcompressor.app.diffusion.config import DiffusionQuantConfig, DiffusionPtqRunConfig


from deepcompressor.nn.patch.lowrank import LowRankBranch # Make sure to import this

# This script now requires the nunchaku._C library to be compiled and installed.
import nunchaku._C as nunchaku_C

def ceil_div(a, b):
    return (a + b - 1) // b

class SVDQuantLinear(nn.Module):
    """
    Python wrapper for the Nunchaku W4A4+LoRA fused GEMM kernel.
    """
    def __init__(self, in_features, out_features, lora_rank, bias=True, use_fp4=False, device='cuda:0'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.bias_enabled = bias
        self.device = torch.device(device)
        self.dtype = torch.float16

        print("Initializing Nunchaku backend.")
        self.backend = nunchaku_C.QuantizedGEMM()
        
        # Call the C++ init with rank=0 to avoid allocation bugs.
        # The true rank will be set and memory allocated during load_weights.
        self.backend.init(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias_enabled,
            use_fp4=use_fp4,
            bf16=(self.dtype == torch.bfloat16),
            deviceId=self.device.index,
            rank=0 
        )

    def load_weights(self, state_dict):
        # The C++ loadDict function will see the shape of the incoming LoRA tensors
        # and correctly re-allocate the internal memory with the proper rank.
        
        # Handle dtypes correctly. qweight must remain int8.
        processed_dict = {}
        for k, v in state_dict.items():
            if k == 'qweight':
                # Do not change the dtype of qweight
                processed_dict[k] = v.to(self.device).contiguous()
            else:
                processed_dict[k] = v.to(self.device, self.dtype).contiguous()

        self.backend.loadDict(processed_dict, False)
        print("Weights loaded into the backend.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = x.to(self.device, self.dtype).contiguous()
        # The backend expects padded inputs.
        #in_features_pad = ceil_div(self.in_features, 128) * 128
        
        #if x.shape[-1] != in_features_pad:
        #    x_padded = F.pad(x, (0, in_features_pad - self.in_features))
        #else:
        #    x_padded = x

        output_padded = self.backend.forward(x)
        # Slice the output back to the original feature dimension
        #return output_padded[:, :self.out_features]
        return output_padded

# --- Main Debugging Logic ---

# 1. Load the "black box" golden model and all artifacts
print("--- Loading Golden Model and All Artifacts ---")
golden_model = torch.load('../deepcompressor/runs/diffusion/int4_rank32_batch12/model/golden_reference.pkl', weights_only=False).eval()
base_path = '../deepcompressor/runs/diffusion/int4_rank32_batch12/model/'
weights_state_dict = torch.load(os.path.join(base_path, 'model.pt'))
scale_state_dict = torch.load(os.path.join(base_path, 'scale.pt'))
branch_state_dict = torch.load(os.path.join(base_path, 'branch.pt'))
smooth_scales = torch.load(os.path.join(base_path, 'smooth.pt'))
configs, _, _, _, _ = DiffusionPtqRunConfig.get_parser().parse_known_args()
config = configs.quant

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16


# 2. Choose a target layer to test
# Let's pick the first FFN's input layer for simplicity.
layer_name = "block_chunks.6.module.2.ffn.fc1"
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

#Check Scale
weight_target = weights_state_dict[f"{layer_name}.weight"]
scale = scale_state_dict[f"{layer_name}.weight.scale.0"]

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
quantizer = Quantizer(config.ipts, key=layer_name, channels_dim=-1)
quantizer.smooth_scale = smooth_scales[layer_name] # Configure with the scale!
quantizer.input_packager = SimpleInputPackager()
quantizer.as_hook().register(reconstructed_layer)

# Create a random tensor to run a single forward pass
# For a quick test, a random tensor is often sufficient.
sample_input = torch.randn(
    4,reconstructed_layer.in_features,
    device=device,
    dtype=dtype
)-0.5


# Grab artifacts for this layer
W_fp16   = weights_state_dict[f"{layer_name}.weight"].to(dtype).to(device)          # (out, in)
bias_fp16 = weights_state_dict.get(f"{layer_name}.bias", None)
smooth_vec = smooth_scales[layer_name].to(dtype).to(device)                         # (in,) or (in,1)

# DeepCompressor "dequantization scales" (weight scales)
# Common key pattern: "...weight.scale.0" -> shape (out, 1, groups, 1) or per-tensor
S_scale = scale_state_dict[f"{layer_name}.weight.scale.0"].to(dtype).to(device)

# LoRA branch weights from branch.pt
Ld_plain = branch_state_dict[layer_name]['a.weight'].to(dtype).to(device)           # usually (rank, in) in DC
Lu_plain = branch_state_dict[layer_name]['b.weight'].to(dtype).to(device)           # (out, rank)

# Ensure Ld is (rank, in_features) for the converter
rank = Lu_plain.shape[1]
if Ld_plain.shape[0] != rank:
    # then it's likely (in, rank) -> transpose
    Ld_conv = Ld_plain.T.contiguous()
else:
    Ld_conv = Ld_plain.contiguous()

print('--------- Before Conversion -----------')
print('In features: ', target_golden_layer.in_features)
print('In features: ', target_golden_layer.out_features)
print('W: ', W_fp16.shape)
print('Bias: ', bias_fp16.shape)
print('Scale:', S_scale.shape)
print('Smooth Vec: ', smooth_vec.shape)
print('Lora Down: ', Ld_conv.shape)
print('Lora Up: ', Lu_plain.shape)

# Use the official converter to PACK everything exactly as the CUDA kernel expects.
# Important: smooth_fused=False so the converter "unsmooths" LoRA-down (divides by smooth),
# matching your ActivationSmoother hook that multiplies inputs by 'smooth'.
state = convert_to_nunchaku_w4x4y16_linear_state_dict(
    weight=W_fp16,            # residual weights (fp16)
    scale=S_scale,            # DC dequant scales (same ones the runtime should use)
    bias=bias_fp16,           # or None
    smooth=smooth_vec,        # 1D smooth vector
    lora=(Ld_conv, Lu_plain), # Ld (rank, in), Lu (out, rank) in PLAIN fp16; converter packs both
    smooth_fused=False,       # critical for parity with your hooks
    float_point=False,
    subscale=None,
)

print('--------- After Conversion -----------')
for k,v in state.items():
    print(k, ' ', v.shape)

# The converter returns a ready-to-load dict with:
#   qweight, (wscales OR wcscales/wtscale depending on granularity), bias, smooth, lora_down, lora_up (packed)
svd_linear = SVDQuantLinear(
    in_features=target_golden_layer.in_features,
    out_features=target_golden_layer.out_features,
    lora_rank=rank,
    bias=(bias_fp16 is not None),
    use_fp4=False,
    device=device,
)
svd_linear.load_weights(state)
print('Weights Loaded correctly (packed by converter)')

# 5. Pass the captured input through the reconstructed layer
print("--- Running Inference on Reconstructed Layer ---")
with torch.no_grad():
    reconstructed_output = reconstructed_layer(sample_input.float())
    cpp_output = svd_linear(sample_input).float()

# 6. The Definitive Comparison
print("\n--- Verification ---")

are_outputs_close = torch.allclose(cpp_output, reconstructed_output, atol=1e-3)
print(f"Does the output of the golden layer match the reconstructed layer? -> {are_outputs_close}")
if not are_outputs_close:
    print(f"Max difference: {(cpp_output - reconstructed_output).abs().max().item()}")
    print(f"Mean absolute difference: {(cpp_output - reconstructed_output).abs().mean().item()}")
    mse = torch.mean((reconstructed_output - cpp_output).float() ** 2)
    print(f"Mean Squared Error (MSE) between ideal and SVDQuant layer: {mse.item():.6f}")
    snr = 10 * torch.log10(torch.mean(reconstructed_output.float()**2) / mse).item()
    print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")
    print("Mean Absolute Value of output:", cpp_output.abs().mean())


# Flatten to a single vector (global cosine similarity)
cos_sim_global = F.cosine_similarity(
    cpp_output.flatten(), 
    reconstructed_output.flatten(), 
    dim=0
).item()

# Row-wise (per-sample) cosine similarity, then mean
cos_sim_per_sample = F.cosine_similarity(
    cpp_output, 
    reconstructed_output, 
    dim=1
).mean().item()

print(f"Global cosine similarity: {cos_sim_global:.6f}")
print(f"Mean per-sample cosine similarity: {cos_sim_per_sample:.6f}")

