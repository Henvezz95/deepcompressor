import torch
import os
from deepcompressor.calib.smooth import ActivationSmoother
from deepcompressor.utils.hooks import ProcessHook

# --- Configuration ---
ARTIFACT_DIR = 'runs/diffusion/int4_rank32_batch12/model/'
PKL_PATH = os.path.join(ARTIFACT_DIR, 'golden_reference.pkl')

print(f"--- Inspecting Golden Reference for Active ActivationSmoother Hooks ---")
print(f"Loading model from: {PKL_PATH}\n")

# --- Load the golden reference model ---
try:
    golden_model = torch.load(PKL_PATH, weights_only=False).eval()
    
    found_hooks = 0
    
    # Iterate through all named modules in the model
    for name, module in golden_model.named_modules():
        # Check if the module has forward hooks attached
        if hasattr(module, '_forward_pre_hooks') and module._forward_pre_hooks:
            # Iterate through all hooks on this module
            for hook in module._forward_pre_hooks.values():
                # Check if the hook is an ActivationSmoother instance
                if isinstance(hook, ActivationSmoother) or isinstance(hook, ProcessHook):
                    found_hooks += 1
                    
                    print(f"--- Hook #{found_hooks} Found ---")
                    print(f"Layer Name: {name}")

                        
                    # Print stats about the smoothing scale used by the hook
                    if isinstance(hook, ProcessHook):
                        scale = hook.processor.smooth_scale
                    else:
                        scale = hook.smooth_scale
                    print(f"Weight Shape: {scale.shape}")
                    print(f"Smoother Scale Stats: Min={scale.min():.4f}, Max={scale.max():.4f}, Mean={scale.mean():.4f}")
                    print("-" * 25)

    if found_hooks == 0:
        print("\nNo active ActivationSmoother hooks were found in the model.")
    else:
        print(f"\n✅ Inspection Complete. Found {found_hooks} active ActivationSmoother hooks.")

except FileNotFoundError:
    print(f"❌ ERROR: Could not find the file: {PKL_PATH}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")