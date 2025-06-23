# deepcompressor/app/diffusion/dataset/infinity_calib_loader.py

import torch
import torch.nn as nn
import os
import re
from tqdm import tqdm
from collections import defaultdict
import typing as tp
import gc
from collections import defaultdict, OrderedDict
from functools import partial

from deepcompressor.data.cache import IOTensorsCache, TensorsCache, TensorCache
from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct, PatchedCrossAttention
from deepcompressor.app.diffusion.nn.struct import DiffusionBlockStruct

class InfinityCalibManager:
    """
    This optimized version correctly handles the stateful nature of the
    Infinity model by iterating through prompts within a per-block loop.
    It loads the full history for one prompt at a time to dramatically reduce
    redundant file I/O during KV cache reconstruction.
    """
    def __init__(self, model: InfinityStruct, cache_dir: str, batch_size: int):
        self.model_struct = model
        self.cache_dir = cache_dir
        
        self.prompts = defaultdict(dict)
        filepaths = [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir) if f.endswith(".pt")]
        
        pattern = re.compile(r"(.+)_step_(\d+)\.pt$")
        for filepath in filepaths:
            match = pattern.search(os.path.basename(filepath))
            if not match: continue
            prompt_filename, step_str = match.groups()
            self.prompts[prompt_filename][int(step_str)] = filepath
            
        print(f"InfinityCalibManager initialized for {len(self.prompts)} prompts.")

    def _set_kv_cache_for_step(self, model_instance, step_index, prompt_filename):
        """Helper to reconstruct and set the KV cache for a single step."""
        device = next(model_instance.parameters()).device
        
        def find_past_filepath(p_filename, p_step):
            return os.path.join(self.cache_dir, f"{p_filename}_step_{p_step:02d}.pt")

        for name, mod in model_instance.named_modules():
            if hasattr(mod, 'kv_caching'):
                if step_index == 0:
                    mod.cached_k, mod.cached_v = None, None
                else:
                    history_k, history_v = [], []
                    for prev_step in range(step_index):
                        past_filepath = find_past_filepath(prompt_filename, prev_step)
                        if os.path.exists(past_filepath):
                            try:
                                past_state = torch.load(past_filepath, map_location='cpu')
                                if name in past_state.get('kv_deltas', {}):
                                    history_k.append(past_state['kv_deltas'][name]['k_delta'])
                                    history_v.append(past_state['kv_deltas'][name]['v_delta'])
                            except (FileNotFoundError, EOFError):
                                continue
                    
                    if history_k:
                        mod.cached_k = torch.cat(history_k, dim=2).to(device)
                        mod.cached_v = torch.cat(history_v, dim=2).to(device)
                    else:
                        mod.cached_k, mod.cached_v = None, None

    def _collect_activations_for_block(self, block_struct: DiffusionBlockStruct):
        """
        Performs a pass over all prompts to collect activations for a single block.
        """
        device = next(self.model_struct.module.parameters()).device
        aggregated_activations = defaultdict(list)
        representative_block_kwargs = {}
        
        for prompt_filename, steps in tqdm(self.prompts.items(), desc=f"Processing Prompts for {block_struct.name}", leave=False):
            prompt_history = {step: torch.load(path, map_location='cpu') for step, path in steps.items()}
            for step_index in sorted(prompt_history.keys()):
                state_dict = prompt_history[step_index]
                self._set_kv_cache_for_step(self.model_struct.module, step_index, prompt_filename)
                with torch.no_grad():
                    hidden_states = state_dict['input_hidden_states'].to(device).float()
                    block_kwargs = {
                        'cond_BD': state_dict['cond_BD'].to(device),
                        'ca_kv': tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in state_dict['ca_kv']),
                        'scale_schedule': state_dict['scale_schedule'], 'scale_ind': state_dict['scale_ind'],
                        'rope2d_freqs_grid': state_dict['rope2d_freqs_grid'], 'attn_bias_or_two_vector': None,
                    }
                    if not representative_block_kwargs:
                        representative_block_kwargs = block_kwargs
                    
                    hooks = []
                    def get_hook(key):
                        def hook_fn(module, input_tuple, output):
                            # For CrossAttention, we only capture hidden_states (args[0]).
                            if isinstance(module, PatchedCrossAttention):
                                if input_tuple and isinstance(input_tuple[0], torch.Tensor):
                                    aggregated_activations[key].append(input_tuple[0].detach().cpu())
                            # For all other modules, capture all tensor inputs.
                            else:
                                for arg in input_tuple:
                                    if isinstance(arg, torch.Tensor):
                                        aggregated_activations[key].append(arg.detach().cpu())
                        return hook_fn

                    # 1. Hook the individual linear layers
                    for _, module_name, module, _, _ in block_struct.named_key_modules():
                        if isinstance(module, nn.Linear):
                            hooks.append(module.register_forward_hook(get_hook(module_name)))
                    
                    # 2. ALSO hook the container modules (Attention and FFN)
                    submodule_structs_to_process = block_struct.attn_structs + [block_struct.ffn_struct]
                    for sub_mod_struct in submodule_structs_to_process:
                        if sub_mod_struct:
                            module_name = sub_mod_struct.name
                            module_to_hook = sub_mod_struct.module
                            hooks.append(module_to_hook.register_forward_hook(get_hook(module_name)))

                    hidden_states = block_struct.module(hidden_states, **block_kwargs)

                    for h in hooks:
                        h.remove()
        
        final_layer_cache = {}
        # Iterate through the modules we know exist in the block to package the cache.
        for _, module_name, module, _, _ in block_struct.named_key_modules():
            if module_name in aggregated_activations:
                tensor_list = aggregated_activations[module_name]
                if tensor_list:
                    channels_dim = -1 if isinstance(module, nn.Linear) else 1
                    tensor_cache = TensorCache(data=tensor_list, channels_dim=channels_dim)
                    tensors_cache = TensorsCache(tensors=OrderedDict([(0, tensor_cache)]))
                    final_layer_cache[module_name] = IOTensorsCache(inputs=tensors_cache)
        
        # Also package the container module activations
        submodule_structs_to_process = block_struct.attn_structs + [block_struct.ffn_struct]
        for sub_mod_struct in submodule_structs_to_process:
            if sub_mod_struct and sub_mod_struct.name in aggregated_activations:
                tensor_list = aggregated_activations[sub_mod_struct.name]
                if tensor_list:
                    channels_dim = -1 # Container inputs are typically (B, L, C)
                    tensor_cache = TensorCache(data=tensor_list, channels_dim=channels_dim)
                    tensors_cache = TensorsCache(tensors=OrderedDict([(0, tensor_cache)]))
                    final_layer_cache[sub_mod_struct.name] = IOTensorsCache(inputs=tensors_cache)

        return final_layer_cache, representative_block_kwargs

    def iter_layer_activations(self):
        """
        The main generator. It iterates through each block, triggers the full
        aggregation for it, and then yields the complete cache for that block.
        """
        device = next(self.model_struct.module.parameters()).device
        for block_struct in self.model_struct.iter_transformer_block_structs():
            # Collects the cache with all tensors on the CPU
            cpu_aggregated_cache, block_kwargs = self._collect_activations_for_block(block_struct)
            
            # Create a new cache, moving all tensors to the correct GPU device
            gpu_aggregated_cache = {}
            for module_name, iot_cache in cpu_aggregated_cache.items():
                if iot_cache.inputs:
                    gpu_tensors_dict = OrderedDict()
                    for key, tensor_cache in iot_cache.inputs.items():
                        # Create a new list with tensors moved to the GPU
                        gpu_data = [t.to(device) for t in tensor_cache.data]
                        # Create a new TensorCache with the GPU data
                        gpu_tensors_dict[key] = TensorCache(
                            data=gpu_data,
                            channels_dim=tensor_cache.channels_dim,
                            orig_device=device # Update the original device metadata
                        )
                    gpu_aggregated_cache[module_name] = IOTensorsCache(inputs=TensorsCache(tensors=gpu_tensors_dict))

            yield block_struct, gpu_aggregated_cache, block_kwargs
            
            # Clean up memory before processing the next block
            del cpu_aggregated_cache, gpu_aggregated_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
