import os, gc
import cv2
import torch
from torch import nn
import argparse
import numpy as np
from collections import OrderedDict
import sys
sys.path.append('../')
import omniconfig
from deepcompressor.utils import tools
import json

# Assuming all your custom and library imports are correctly set up
# (Imports from previous files are included here for completeness)
from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct, patchModel, DiffusionAttentionStruct
from deepcompressor.app.diffusion.quant.weight import calibrate_diffusion_block_low_rank_branch
from deepcompressor.calib.smooth import ActivationSmoother
from deepcompressor.quantizer import Quantizer
from deepcompressor.utils.hooks import SimpleInputPackager
from deepcompressor.app.diffusion.nn.struct import DiTStruct
from deepcompressor.app.diffusion.config import DiffusionQuantConfig, DiffusionPtqRunConfig, DiffusionEvalConfig


from Infinity_rep.infinity.models.infinity import Infinity
from Infinity_rep.tools.run_infinity import load_visual_tokenizer, load_tokenizer, gen_one_img, load_transformer
from Infinity_rep.infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from deepcompressor.nn.patch.lowrank import LowRankBranch # Make sure to import this
# --- WARNING: This script is for debugging only. Loading pickled models is not recommended for production. ---
from PIL import Image

class InfinityPipelineWrapper:
    def __init__(self, model, vae, tokenizer, text_encoder, generation_args):
        self.quantized_model = model
        self.vae = vae
        self.text_tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.generation_args = generation_args
        self.scheduler = 'Custom' # Dummy attribute for compatibility

    def to(self, device):
        self.quantized_model.to(device)
        return self
    
    def set_progress_bar_config(self, **kwargs):
        # This is a dummy method to satisfy the evaluation framework's API.
        # It doesn't need to do anything.
        pass

    def __call__(self, prompt: list[str], generator: list[torch.Generator], **kwargs):
        # The __call__ should handle a batch of prompts
        pil_images = []
        for i, p in enumerate(prompt):
            # Update the seed for each image in the batch
            self.generation_args['g_seed'] = generator[i].initial_seed()
            
            # Generate one image
            image_tensor = gen_one_img(
                self.quantized_model, self.vae, self.text_tokenizer, 
                self.text_encoder, p, **self.generation_args
            )
            # Convert to the expected PIL format
            img = cv2.cvtColor(image_tensor.detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(img))
            # --- ADD THESE LINES TO FORCE MEMORY CLEANUP ---
            gc.collect()
            torch.cuda.empty_cache()
        
        # The framework expects an object with an 'images' attribute
        class PipelineOutput:
            def __init__(self, images):
                self.images = images
        
        return PipelineOutput(pil_images)

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

logger = tools.logging.getLogger(__name__)
ptq_config, _, unused_cfgs, unused_args, unknown_args = DiffusionPtqRunConfig.get_parser().parse_known_args()

#  Extract the config objects as before
quant_config = ptq_config.quant
eval_config = ptq_config.eval

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

for module_key, module_name, module, _, _ in model_struct.named_key_modules():
    # Check if this layer should have its activations quantized
    if quant_config.ipts.is_enabled_for(module_key):
        print(f"Attaching ACTIVE A4 Quantizer hook to: {module_name}")
        quantizer = Quantizer(quant_config.ipts, key=module_name, channels_dim=-1)
        
        # The crucial fix:
        quantizer.input_packager = SimpleInputPackager()
        quantizer.as_hook().register(module)

print("✅ Activation hooks are now truly active.")

prompt = 'An idillic place, with grass, water and beautiful colors in a relaxing atmosphere'
save_name = "_".join(prompt.split(' '))

generation_args = { 'cfg_list': [3.0]*13, 'tau_list': [0.5]*13, 'g_seed': 16,
                    'gt_leak': 0, 'gt_ls_Bl': None, 'scale_schedule': scale_schedule,
                    'cfg_insertion_layer': [args.cfg_insertion_layer], 'vae_type': args.vae_type,
                    'sampling_per_bits': args.sampling_per_bits, 'enable_positive_prompt': False }
#infinity_pipeline = InfinityPipelineWrapper(quantized_model, vae, text_tokenizer, text_encoder, generation_args)
infinity_pipeline = InfinityPipelineWrapper(model, vae, text_tokenizer, text_encoder, generation_args)

results = eval_config.evaluate(infinity_pipeline, skip_gen=False, task=ptq_config.pipeline.task)

logger.info(f"* Saving results to {ptq_config.output.job_dirpath}")
with open(ptq_config.output.get_running_job_path("results.json"), "w") as f:
    json.dump(results, f, indent=2, sort_keys=True)