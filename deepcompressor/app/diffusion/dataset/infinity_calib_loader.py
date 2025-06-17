# infinity_calib_manager.py

import torch
import torch.nn as nn
import os
from tqdm import tqdm
from collections import defaultdict

from deepcompressor.data.cache import (
    IOTensorsCache,
    ModuleForwardInput,
    TensorCache,
    TensorsCache,
)

# Import necessary components from your project
from deepcompressor.app.diffusion.nn.struct_infinity import Infinity, DiTStruct, CrossAttnBlock
from torch.utils.data import Dataset

class InfinityStatefulCalibDataset(Dataset):
    """
    A custom Dataset class to load and organize the stateful cache created
    by our 'Writer' script. It intelligently groups the captured states by
    layer name and autoregressive scale index.
    """
    def __init__(self, cache_dir: str, model_struct: DiTStruct):
        super().__init__()
        self.data_by_layer_and_scale = defaultdict(lambda: defaultdict(list))
        self.layer_names = [name for name, _ in model_struct.named_modules()]

        print(f"Loading and indexing stateful cache from: {cache_dir}")
        filepaths = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".pt")]
        
        for filepath in tqdm(filepaths, desc="Indexing cache files"):
            # Each file contains the history for one image generation
            generation_history = torch.load(filepath, map_location='cpu')
            
            for step_state in generation_history:
                scale_ind = step_state['scale_ind']
                
                # For this step, add an entry for every layer in the model.
                # The data is the same for all layers, but this structure makes
                # retrieval in the manager class much simpler.
                for layer_name in self.layer_names:
                    self.data_by_layer_and_scale[layer_name][scale_ind].append(step_state)
        
        # Create a flat list for __getitem__ and __len__
        self._flat_data = []
        for layer_name in self.data_by_layer_and_scale:
            for scale_ind in self.data_by_layer_and_scale[layer_name]:
                for item in self.data_by_layer_and_scale[layer_name][scale_ind]:
                    self._flat_data.append(item)

    def __len__(self):
        return len(self._flat_data)

    def __getitem__(self, idx):
        return self._flat_data[idx]

    def get_data_for_layer_and_scale(self, layer_name, scale_ind):
        return self.data_by_layer_and_scale.get(layer_name, {}).get(scale_ind, [])


class InfinityCalibManager:
    """
    This class replaces the default `iter_layer_activations`. It provides a
    generator that yields perfectly prepared, stateful, and homogeneous batches
    of data for each layer at each scale, which the quantization algorithms can consume.
    """
    def __init__(self, model: Infinity, cache_dir: str, batch_size: int):
        self.model = model
        self.model_struct = DiTStruct.construct(model)
        self.batch_size = batch_size
        self.dataset = InfinityStatefulCalibDataset(cache_dir, self.model_struct)

    def collate_fn(self, batch_list: list[dict]):
        """
        Custom collate function to stack the dictionaries of tensors into a single batch dictionary.
        """
        if not batch_list:
            return None
        
        # Collate simple tensors and other data types
        collated = {key: [d[key] for d in batch_list] for key in batch_list[0] if key not in ['kv_deltas']}
        collated['input_hidden_states'] = torch.stack(collated['input_hidden_states'])
        collated['cond_BD'] = torch.stack(collated['cond_BD'])
        
        # Collate the ca_kv tuple
        ca_kv_list = collated.pop('ca_kv')
        collated['ca_kv'] = (
            torch.cat([item[0] for item in ca_kv_list], dim=0),
            torch.cat([item[1] for item in ca_kv_list], dim=0),
            max([item[2] for item in ca_kv_list])
        )

        # Collate the nested kv_deltas dictionary
        kv_deltas_list = [d['kv_deltas'] for d in batch_list]
        collated_deltas = defaultdict(lambda: defaultdict(list))
        if kv_deltas_list:
            for d in kv_deltas_list:
                for layer_name, tensors in d.items():
                    collated_deltas[layer_name]['k_delta'].append(tensors['k_delta'])
                    collated_deltas[layer_name]['v_delta'].append(tensors['v_delta'])
            
            final_deltas = {}
            for layer_name, tensors in collated_deltas.items():
                final_deltas[layer_name] = {
                    'k_delta': torch.stack(tensors['k_delta']),
                    'v_delta': torch.stack(tensors['v_delta'])
                }
            collated['kv_deltas'] = final_deltas
        else:
            collated['kv_deltas'] = {}
            
        return collated

    def set_kv_cache_from_deltas(self, batched_kv_deltas):
        """
        Reconstructs the KV cache for the current step by iterating through
        all previous steps' deltas and concatenating them.
        """
        # This is a simplified placeholder. A real implementation would need to
        # know the current step index to concatenate deltas from all prior steps.
        # For now, we assume this function correctly sets the model's KV cache.
        for layer_name, deltas in batched_kv_deltas.items():
            module = dict(self.model.named_modules())[layer_name]
            module.cached_k = deltas['k_delta']
            module.cached_v = deltas['v_delta']

    def iter_for_quantization(self):
        """
        The main generator function. This REPLACES the framework's default
        `iter_layer_activations` function.
        """
        device = next(self.model.parameters()).device
        
        # Iterate through each transformer block in the model
        for block_struct in tqdm(self.model_struct.iter_transformer_block_structs(), desc="Blocks"):
            
            # For each block, get all its submodules (sa, ca, ffn)
            for submodule_name, submodule in block_struct.iter_submodules():
                layer_name = f"{block_struct.rname}.{submodule_name}"
                
                # For each submodule, iterate through the autoregressive scales we have data for
                for scale_ind in sorted(self.dataset.data_by_layer_and_scale.get(layer_name, {}).keys()):
                    
                    scale_data = self.dataset.get_data_for_layer_and_scale(layer_name, scale_ind)
                    
                    # Create mini-batches from this homogeneous data
                    for i in range(0, len(scale_data), self.batch_size):
                        batch_list = scale_data[i: i + self.batch_size]
                        
                        # Collate the list of dicts into a single batch dict
                        batch_dict = self.collate_fn(batch_list)
                        if batch_dict is None: continue

                        # --- Replay the forward pass to get correct internal inputs ---
                        with torch.no_grad():
                            # 1. Reconstruct and set the KV cache for this step
                            #    (A more advanced version would concat all prior deltas)
                            self.set_kv_cache_from_deltas(batch_dict['kv_deltas'])

                            # 2. Prepare all inputs for the block's forward pass
                            block_input = batch_dict['input_hidden_states'].to(device)
                            block_kwargs = {
                                'cond_BD': batch_dict['cond_BD'].to(device),
                                'ca_kv': tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch_dict['ca_kv']),
                                'scale_schedule': batch_dict['scale_schedule'][0], # Same for all in batch
                                'scale_ind': batch_dict['scale_ind'][0],
                                'rope2d_freqs_grid': batch_dict['rope2d_freqs_grid'][0]
                            }

                            # 3. Create a temporary cache to capture inputs for each submodule
                            temp_cache = {}
                            hooks = []
                            
                            def get_hook(name):
                                def hook_fn(module, input, output):
                                    # Create an IOTensorsCache on the fly
                                    temp_cache[name] = IOTensorsCache(
                                        inputs=TensorsCache.from_dict({'hidden_states': input[0].detach().cpu()}),
                                        outputs=TensorCache(output.detach().cpu())
                                    )
                                return hook_fn

                            for name, mod in block_struct.module.named_modules():
                                if name in ['sa', 'ca', 'ffn']:
                                    hooks.append(mod.register_forward_hook(get_hook(name)))
                            
                            # 4. Run the forward pass for the *entire block*
                            block_struct.module(block_input, **block_kwargs)
                            
                            # 5. Remove hooks
                            for h in hooks:
                                h.remove()
                        
                        # 6. Yield the captured data for the target submodule
                        yield (
                            layer_name,
                            (
                                block_struct,  # The struct for the whole block
                                temp_cache,    # The cache with inputs/outputs for each submodule
                                {}             # Empty kwargs
                            )
                        )
