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
    def __init__(self, model: InfinityStruct, cache_dir: str, batch_size: int, smooth_cache: dict[str, torch.Tensor] | None = None):
        self.model_struct = model
        self.cache_dir = cache_dir
        if not smooth_cache or smooth_cache is None:
            self.skip = False
        else:
            self.skip = True

        # Set all self-attention modules to stateless mode once during initialization.
        print("Setting self-attention modules to stateless mode (caching=False) for calibration.")
        for mod in self.model_struct.module.modules():
            # Check for the attribute that identifies the self-attention modules
            # This could be 'kv_caching' or a more specific class type.
            if hasattr(mod, 'caching'): 
                mod.caching = False
        
        self.prompts = defaultdict(dict)
        filepaths = [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir) if f.endswith(".pt")]
        
        pattern = re.compile(r"(.+)_step_(\d+)\.pt$")
        for filepath in filepaths:
            match = pattern.search(os.path.basename(filepath))
            if not match: continue
            prompt_filename, step_str = match.groups()
            self.prompts[prompt_filename][int(step_str)] = filepath
            
        print(f"InfinityCalibManager initialized for {len(self.prompts)} prompts.")


    def _build_sa_kv_cache(self, step_index, prompt_step_history, block_struct):
        """
        Builds and returns a dictionary of cumulative SA KV caches for a given step.
        This function is purely stateless and does not modify the model.
        """
        sa_kv_cache = {}
        device = next(self.model_struct.module.parameters()).device

        # 1. Iterate through the block's submodules to find the self-attention container.
        #    The 'attn_structs' property is the correct place to look.
        for attn_struct in block_struct.attn_structs:
            # 2. Check if this is a self-attention module (not cross-attention).
            #    We can identify it by the 'kv_caching' attribute we know it has.
            if hasattr(attn_struct.module, 'caching'):
                local_sa_name = attn_struct.name.split('.')[-1]  # This will be the local name, e.g., 'sa'
                
                # 3. Construct the fully-qualified name that was used as the key when saving.
                #    This combines the block's global name with the submodule's local name.
                expected_key_in_cache = f"{block_struct.name}.{local_sa_name}"

                if step_index > 0:
                    history_k, history_v = [], []
                    for prev_step in range(step_index):
                        past_state = prompt_step_history[prev_step]
                        
                        # 4. Use the correctly constructed key for the lookup in the cache file.
                        if expected_key_in_cache in past_state.get('kv_deltas', {}):
                            delta = past_state['kv_deltas'][expected_key_in_cache]
                            history_k.append(delta['k_delta'])
                            history_v.append(delta['v_delta'])

                    if history_k:
                        cumulative_k = torch.cat(history_k, dim=2).to(device)
                        cumulative_v = torch.cat(history_v, dim=2).to(device)
                        
                        # 5. Store the cache in our dictionary using the LOCAL name ('sa'),
                        #    which is what the PatchedSelfAttention module will look for in kwargs.
                        sa_kv_cache[local_sa_name] = {'k': cumulative_k, 'v': cumulative_v}      
        return sa_kv_cache

    def _collect_activations_for_block(self, block_struct: DiffusionBlockStruct):
        """
        Performs a pass over all prompts to collect activations for a single block.
        """
        device = next(self.model_struct.module.parameters()).device
        aggregated_activations = defaultdict(list)
        representative_block_kwargs = {}
        for prompt_filename, steps in tqdm(self.prompts.items(), desc=f"Processing Prompts for {block_struct.name}", leave=False):
    
            # Now 'steps' is defined, and this line will work correctly
            prompt_step_history = {step: torch.load(path, map_location='cpu') for step, path in steps.items()}
            
            for step_index in sorted(prompt_step_history.keys()):
                state_dict = prompt_step_history[step_index]
                cumulative_sa_cache = self._build_sa_kv_cache(step_index, prompt_step_history, block_struct)
                with torch.no_grad():
                    hidden_states = state_dict['input_hidden_states'].to(device).float()
                    block_kwargs = {
                        'cond_BD': state_dict['cond_BD'].to(device),
                        'ca_kv': tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in state_dict['ca_kv']),
                        'scale_schedule': state_dict['scale_schedule'], 'scale_ind': state_dict['scale_ind'],
                        'rope2d_freqs_grid': state_dict['rope2d_freqs_grid'], 'attn_bias_or_two_vector': None,
                        'sa_kv_cache': cumulative_sa_cache
                    }
                    if not representative_block_kwargs:
                        representative_block_kwargs = block_kwargs
                    
                    hooks = []
                    def get_hook(key, module): # Pass the actual module instance
                        # The default hook for most layers
                        def default_hook_fn(module, input_tuple, output):
                            for arg in input_tuple:
                                if isinstance(arg, torch.Tensor):
                                    aggregated_activations[key].append(arg.detach().cpu())

                        # A special hook for our attention block containers
                        def container_hook_fn(module, input_tuple, output):
                            # Capture positional inputs like hidden_states
                            for arg in input_tuple:
                                if isinstance(arg, torch.Tensor):
                                    aggregated_activations[key].append(arg.detach().cpu())
                            
                            # Also capture our special 'sa_kv_cache' from the keyword arguments
                            # kwargs are passed as the last element of the input tuple if they exist
                            if input_tuple and isinstance(input_tuple[-1], dict):
                                kwargs = input_tuple[-1]
                                if 'sa_kv_cache' in kwargs and kwargs['sa_kv_cache']:
                                    # Save this data under a special key so the consumer knows what it is.
                                    # The key could be the block name itself, as it's the main "activation".
                                    aggregated_activations[key].append(kwargs['sa_kv_cache'])

                        # Return the correct hook based on the module type
                        if isinstance(module, PatchedCrossAttention): # Or your new CalibCrossAttnBlock
                            return container_hook_fn
                        else:
                            return default_hook_fn

                    # 1. Hook the individual linear layers
                    for _, module_name, module, _, _ in block_struct.named_key_modules():
                        if isinstance(module, nn.Linear):
                            hooks.append(module.register_forward_hook(get_hook(module_name, module)))
                    
                    # 2. ALSO hook the container modules (Attention and FFN)
                    submodule_structs_to_process = block_struct.attn_structs + [block_struct.ffn_struct]
                    for sub_mod_struct in submodule_structs_to_process:
                        if sub_mod_struct:
                            module_name = sub_mod_struct.name
                            module_to_hook = sub_mod_struct.module
                            hooks.append(module_to_hook.register_forward_hook(get_hook(module_name, module_to_hook)))

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
            if self.skip:
                yield block_struct, {}, {}
                continue
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
