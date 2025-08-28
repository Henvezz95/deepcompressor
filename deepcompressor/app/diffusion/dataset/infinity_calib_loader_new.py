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
import datasets
from omniconfig import configclass
from dataclasses import dataclass

from deepcompressor.data.cache import IOTensorsCache, TensorsCache, TensorCache
from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct, PatchedCrossAttention
from deepcompressor.app.diffusion.nn.struct import DiffusionBlockStruct
from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.app.diffusion.dataset.collect.online_infinity_generation import get_stateful_cache
from deepcompressor.app.diffusion.dataset.collect.calib import CollectConfig
from deepcompressor.app.diffusion.dataset.data import get_dataset


class InfinityCalibManager:
    """
    Online input generator for Stateful Infinity Model
    """
    def __init__(self, model: InfinityStruct, config: DiffusionPtqRunConfig, 
                 other_configs: dict, smooth_cache: dict[str, torch.Tensor] | None = None,
                 save_kv_cache_only: bool = False, save_imgs: bool = True, skip_keys = []):
        self.model_struct = model
        self.config = config
        
        collect_config, pipeline_config = other_configs["collect"], other_configs["pipeline"]
        dataset = get_dataset(
            collect_config['data_path'],
            max_dataset_size=collect_config['num_samples'],
            repeat=1,
        )
        self.dataset = dataset
        self.pipeline_config = pipeline_config
        self.save_kv_cache_only = save_kv_cache_only
        self.save_imgs = save_imgs
        self.skip_keys = skip_keys

        if not smooth_cache or smooth_cache is None:
            self.skip = False
        else:
            self.skip = True

    def iter_layer_activations(self):
        """
        The main generator. It iterates through each block, triggers the full
        aggregation for it, and then yields the complete cache for that block.
        """
        device = next(self.model_struct.module.parameters()).device
        for block_struct in self.model_struct.iter_transformer_block_structs():
            name = block_struct.name
            if any([k in name for k in self.skip_keys]):
                yield block_struct, {}, {}
                continue

            # 1. Get cache from the required block
            blck_idx = int(name.split('.')[1])
            mod_idx = int(name.split('.')[3])
            self.model_struct.module.set_block(blck_idx, mod_idx)

            collected_data_list = get_stateful_cache(self.model_struct.module, self.config, self.pipeline_config, 
                                                     self.dataset, blck_idx, mod_idx, save_kv_cache_only=self.save_kv_cache_only, save_imgs=self.save_imgs)
            
            # 2. Initialize structures to hold the re-packaged data
            aggregated_inputs = defaultdict(list)
            block_kwargs_list = []

            # 3. Process the raw collected data
            for step_data in collected_data_list:
                # Separate the evaluation kwargs from the direct linear inputs
                if 'eval_kwargs' in step_data:
                    block_kwargs_list.append(step_data.pop('eval_kwargs'))
                
                # Aggregate the direct inputs for each layer
                for module_name, tensor in step_data.items():
                    aggregated_inputs[module_name].append(tensor.cpu())

            # 4. Package the aggregated inputs into the IOTensorsCache format
            final_aggregated_cache = {}

            # 5. Define a clear mapping from the simple hook names to the required module name suffixes
            key_to_suffix_map = {
                'sa': '.sa',
                'ca': '.ca', 
                'sa_q': '.sa.to_q',
                'sa_k': '.sa.to_k',
                'sa_v': '.sa.to_v',
                'sa_out': '.sa.proj',
                'ca_q': '.ca.to_q',
                'ca_k': '.ca.to_k',
                'ca_v': '.ca.to_v',
                'ca_out': '.ca.proj',
                'ffn_fc1': '.ffn.fc1',
                'ffn_fc2': '.ffn.fc2',
                'sa_k_final': '.sa.k.cache',
                'sa_v_final': '.sa.v.cache'
            }
            
            base_name = block_struct.name 
            
            for simple_name, tensor_list in aggregated_inputs.items():
                if not tensor_list:
                    continue

                # Look up the suffix from our map
                suffix = key_to_suffix_map.get(simple_name)

                # If the key isn't in our map, print a warning and skip it
                if suffix is None:
                    print(f"Warning: Unhandled key '{simple_name}' in InfinityCalibManager. Skipping.")
                    continue
                
                full_module_name = base_name + suffix
                
                # For linear layers, the channel dimension is the last one
                channels_dim = -1
                # Move tensors to the target device as they are packaged
                gpu_data = [t.to(device).float() for t in tensor_list]
                tensor_cache = TensorCache(data=gpu_data, channels_dim=channels_dim, orig_device=device)
                tensors_cache = TensorsCache(tensors=OrderedDict([(0, tensor_cache)]))
                final_aggregated_cache[full_module_name] = IOTensorsCache(inputs=tensors_cache)

            # 6. Yield the final structures for the current block
            yield block_struct, final_aggregated_cache, block_kwargs_list
            
            # 7. Clean up memory before processing the next block
            del collected_data_list, aggregated_inputs, block_kwargs_list, final_aggregated_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
