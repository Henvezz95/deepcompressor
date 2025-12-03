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

branch_keys = set()
smooth_keys = set()


# 2. Choose a target layer to test
# Let's pick the first FFN's input layer for simplicity.
layer_name = "block_chunks.0.module.0.ffn.fc2"
target_golden_layer = golden_model.get_submodule(layer_name)

# Branch dict
for key in branch_state_dict.keys():
    layer_type = key.split('.')[-1]
    block_type = key.split('.')[-2]
    if layer_type == 'to_q':
        hooks = list(golden_model.get_submodule(key)._forward_pre_hooks.values())
        if block_type == 'sa':
            id = 0
        else:
            id = 1
        
        h_third = branch_state_dict[key]['b.weight'].shape[0]//3
        branch_state_dict[key]['a.weight'] = hooks[id].branch.a.weight
        branch_state_dict[key]['b.weight'][:h_third] = hooks[id].branch.b.weight

        golden_name = key.replace('to_q', 'to_k')
        hooks = list(golden_model.get_submodule(golden_name)._forward_pre_hooks.values())
        branch_state_dict[key]['b.weight'][h_third:2*h_third] = hooks[id].branch.b.weight
        
        golden_name = key.replace('to_q', 'to_v')
        hooks = list(golden_model.get_submodule(golden_name)._forward_pre_hooks.values())
        branch_state_dict[key]['b.weight'][2*h_third:] = hooks[id].branch.b.weight
    elif layer_type == 'proj':
        golden_name = key.replace('proj', 'to_out.0')
        hooks = list(golden_model.get_submodule(golden_name)._forward_pre_hooks.values())
        branch_state_dict[key]['a.weight'] = hooks[1].branch.a.weight
        branch_state_dict[key]['b.weight'] = hooks[1].branch.b.weight
    else:
        hooks = list(golden_model.get_submodule(key)._forward_pre_hooks.values())
        branch_state_dict[key]['a.weight'] = hooks[1].branch.a.weight
        branch_state_dict[key]['b.weight'] = hooks[1].branch.b.weight

    branch_keys.add(key)


# Smooth Scales
for key in smooth_scales.keys():
    layer_type = key.split('.')[-1]
    block_type = key.split('.')[-2]
    if layer_type == 'to_q': 
        if block_type == 'ca':
            hooks = list(golden_model.get_submodule(key)._forward_pre_hooks.values())
            smooth_scales[key] = hooks[0].processor.smooth_scale
        else:
            golden_name = key.replace('.to_q', '')
            hooks = list(golden_model.get_submodule(golden_name)._forward_pre_hooks.values())
            smooth_scales[key] = hooks[0].processor.smooth_scale
    elif layer_type == 'kv_smooth_scale':
        golden_name = key.replace('kv_smooth_scale', 'to_k')
        hooks = list(golden_model.get_submodule(golden_name)._forward_pre_hooks.values())
        smooth_scales[key] = hooks[0].processor.smooth_scale
    elif layer_type == 'proj':
        golden_name = key.replace('proj', 'to_out.0')
        smooth_scales[key] = hooks[0].processor.smooth_scale
    else:
        hooks = list(golden_model.get_submodule(key)._forward_pre_hooks.values())
        smooth_scales[key] = hooks[0].processor.smooth_scale

    smooth_keys.add(key)

assert smooth_keys == set(smooth_scales.keys())
assert branch_keys == set(branch_state_dict.keys())

print('--- Extracted Branch and Smooth Scales from Golden Model ---')