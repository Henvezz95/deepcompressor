# deepcompressor/app/diffusion/dataset/infinity_calib_loader.py

import torch
import torch.nn as nn
import os
import re
from tqdm import tqdm
from collections import defaultdict
import typing as tp
from torch.utils.data import Dataset, DataLoader

# --- FIX: Added ModuleForwardInput to the import list ---
from deepcompressor.data.cache import (
    IOTensorsCache,
    ModuleForwardInput,
    TensorCache,
    TensorsCache,
)

# Import necessary components from your project
from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct
from deepcompressor.app.diffusion.nn.struct import DiffusionBlockStruct, DiffusionModuleStruct

class InfinityStatefulCalibDataset(Dataset):
    """
    A custom Dataset class to load and organize the stateful cache created
    by our 'Writer' script. It organizes all step data by prompt, allowing for
    efficient history reconstruction.
    """
    def __init__(self, cache_dir: str):
        super().__init__()
        # This will store all step data, organized by the original prompt filename.
        # Structure: { "prompt_filename_001": {0: step_0_state, 1: step_1_state, ...}, ... }
        self.histories_by_prompt = defaultdict(dict)
        
        print(f"Loading and indexing stateful cache from: {cache_dir}")
        filepaths = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".pt")]
        
        # Regex to extract prompt filename and step index from the cache file path
        pattern = re.compile(r"(.+)_step_(\d+)\.pt$")

        for filepath in tqdm(filepaths, desc="Indexing cache files"):
            match = pattern.search(os.path.basename(filepath))
            if not match:
                continue
            
            prompt_filename, step_str = match.groups()
            step_index = int(step_str)
            
            step_state = torch.load(filepath, map_location='cpu')
            self.histories_by_prompt[prompt_filename][step_index] = step_state
            
        # Create a flat list of all unique (prompt_filename, step_index) pairs for iteration
        self._flat_lookup = sorted([
            (prompt, step)
            for prompt, history in self.histories_by_prompt.items()
            for step in history.keys()
        ])

    def __len__(self) -> int:
        return len(self._flat_lookup)

    def __getitem__(self, idx: int) -> dict:
        prompt_filename, step_index = self._flat_lookup[idx]
        return {
            'prompt_filename': prompt_filename,
            'step_index': step_index,
            'step_state': self.histories_by_prompt[prompt_filename][step_index]
        }
        
    def get_step_state(self, prompt_filename: str, step_index: int) -> dict | None:
        """Efficiently retrieves the state for a specific prompt and step."""
        return self.histories_by_prompt.get(prompt_filename, {}).get(step_index)


class InfinityCalibManager:
    """
    This class replaces the default `iter_layer_activations`. It provides a
    generator that yields perfectly prepared, stateful, and homogeneous batches
    of data for each layer at each scale, which the quantization algorithms can consume.
    """
    def __init__(self, model: InfinityStruct, cache_dir: str, batch_size: int):
        self.model_struct = model
        self.batch_size = batch_size
        self.dataset = InfinityStatefulCalibDataset(cache_dir)

    def _collate_step_states(self, batch_list: list[dict]) -> dict | None:
        """Custom collate function to stack a list of step_state dictionaries."""
        if not batch_list:
            return None
        
        collated = {}
        first_state = batch_list[0]['step_state']
        
        for key in first_state:
            items_to_collate = [d['step_state'][key] for d in batch_list]
            
            if isinstance(first_state[key], torch.Tensor):
                collated[key] = torch.stack(items_to_collate)
            elif isinstance(first_state[key], tuple) and key == 'ca_kv':
                collated[key] = (
                    torch.cat([item[0] for item in items_to_collate], dim=0),
                    torch.cumsum(torch.tensor([0] + [item[0].shape[0] for item in items_to_collate[:-1]]), dim=0),
                    max([item[2] for item in items_to_collate])
                )
            elif isinstance(first_state[key], dict) and key == 'kv_deltas':
                collated_deltas = defaultdict(lambda: {'k_delta': [], 'v_delta': []})
                for d in items_to_collate:
                    for layer_name, tensors in d.items():
                        collated_deltas[layer_name]['k_delta'].append(tensors['k_delta'])
                        collated_deltas[layer_name]['v_delta'].append(tensors['v_delta'])
                
                collated[key] = {
                    layer_name: {
                        # Concat along the batch dimension (dim=0)
                        'k_delta': torch.cat(tensors['k_delta'], dim=0),
                        'v_delta': torch.cat(tensors['v_delta'], dim=0)
                    } for layer_name, tensors in collated_deltas.items()
                }
            else:
                collated[key] = items_to_collate[0]

        return collated

    def _reconstruct_and_set_kv_cache(self, batch: list[dict], current_scale_ind: int):
        """
        Correctly reconstructs the full KV cache for each item in the batch
        by fetching and concatenating all previous step deltas.
        """
        device = next(self.model_struct.module.parameters()).device
        
        # Get all unique attention layer names from the model structure
        attention_layer_names = [
            f"{block.rname}.sa" for block in self.model_struct.iter_transformer_block_structs()
        ]

        for layer_name in attention_layer_names:
            batch_k_list, batch_v_list = [], []
            for item_context in batch:
                prompt_filename = item_context['prompt_filename']
                
                history_k, history_v = [], []
                for step in range(current_scale_ind):
                    past_state = self.dataset.get_step_state(prompt_filename, step)
                    if past_state and layer_name in past_state['kv_deltas']:
                        history_k.append(past_state['kv_deltas'][layer_name]['k_delta'])
                        history_v.append(past_state['kv_deltas'][layer_name]['v_delta'])
                
                # Concatenate along the sequence dimension (dim=2) for a single prompt
                full_k = torch.cat(history_k, dim=2) if history_k else None
                full_v = torch.cat(history_v, dim=2) if history_v else None
                batch_k_list.append(full_k)
                batch_v_list.append(full_v)

            # Set the reconstructed cache on the live model for the whole batch
            attention_module = self.model_struct.get_module(layer_name)
            if any(k is not None for k in batch_k_list):
                attention_module.cached_k = torch.cat(batch_k_list, dim=0).to(device)
            else:
                attention_module.cached_k = None
            
            if any(v is not None for v in batch_v_list):
                attention_module.cached_v = torch.cat(batch_v_list, dim=0).to(device)
            else:
                attention_module.cached_v = None

    def iter_for_quantization(self):
        """
        The main generator. It efficiently replays the forward pass once per block
        and yields the captured inputs for each submodule.
        """
        device = next(self.model_struct.module.parameters()).device
        
        data_by_scale = defaultdict(list)
        for idx in range(len(self.dataset)):
            data_by_scale[self.dataset._flat_lookup[idx][1]].append(self.dataset[idx])

        for scale_ind in sorted(data_by_scale.keys()):
            scale_specific_data = data_by_scale[scale_ind]
            
            dataloader = DataLoader(scale_specific_data, batch_size=self.batch_size, collate_fn=lambda b: b)

            for batch_list in tqdm(dataloader, desc=f"Scale {scale_ind}", leave=False):
                self._reconstruct_and_set_kv_cache(batch_list, scale_ind)

                batch_dict = self._collate_step_states(batch_list)
                if not batch_dict: continue

                block_input = batch_dict['input_hidden_states'].to(device)
                block_kwargs = {
                    'cond_BD': batch_dict['cond_BD'].to(device),
                    'ca_kv': tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch_dict['ca_kv']),
                    'scale_schedule': batch_dict['scale_schedule'],
                    'scale_ind': batch_dict['scale_ind'],
                    'rope2d_freqs_grid': batch_dict['rope2d_freqs_grid']
                }

                for block_struct in self.model_struct.iter_transformer_block_structs():
                    with torch.no_grad():
                        submodule_inputs_cache = {}
                        hooks = []
                        
                        def get_hook(name):
                            def hook_fn(module, input, output):
                                # --- FIX: Use ModuleForwardInput to capture only inputs ---
                                submodule_inputs_cache[name] = ModuleForwardInput(args=(input[0].detach().cpu(),))
                            return hook_fn

                        for sub_name, sub_mod in block_struct.iter_submodules():
                             # Ensure we only attach to the modules we need
                            if sub_name in ['sa', 'ca', 'ffn']:
                                hooks.append(sub_mod.register_forward_hook(get_hook(sub_name)))
                        
                        block_struct.module(block_input, **block_kwargs)
                        
                        for h in hooks:
                            h.remove()
                    
                    for submodule_name, submodule_struct in block_struct.iter_submodule_structs():
                        if submodule_name in submodule_inputs_cache:
                            # The key for the cache should match the submodule name in iter_submodule_structs
                            layer_cache = {'default': submodule_inputs_cache[submodule_name]}
                            yield (
                                f"{block_struct.rname}.{submodule_name}",
                                (submodule_struct, layer_cache, {})
                            )

