# build_quantized_infinity_for_calib.py
import os, torch, gc
from deepcompressor.app.diffusion.nn.struct_infinity import InfinityStruct, patchModel
from evaluation.build_functions import assemble_model  # same import as in benchmark
from Infinity_rep.tools.run_infinity import load_visual_tokenizer, load_tokenizer
from deepcompressor.app.diffusion.dataset.collect.online_infinity_generation import load_transformer, args_2b, args_8b  # reuse your existing loader/args

def load_quantized_model_for_calib(model_type:str, artifact_dir: str, ptq_config, use_fake_act=True, device="cuda"):
    if '2b' in model_type.lower():
        args = args_2b
    elif '8b' in model_type.lower():
        args = args_8b
    else:
        raise NotImplementedError(f"Model Type not recognized: {model_type}")
    
    # 1) base FP model + patch
    vae = load_visual_tokenizer(args)
    base = load_transformer(vae, args)
    patched = patchModel(base)

    # 2) wrap in InfinityStruct (assemble_model expects a struct, as in benchmark)
    model_struct = InfinityStruct.construct(patched)
    for name, module in patched.named_modules():
        module.name = name

    # 3) load artifacts
    weights = torch.load(os.path.join(artifact_dir, "model.pt"), map_location=device)
    smooth  = torch.load(os.path.join(artifact_dir, "smooth.pt"), map_location=device)
    branch  = torch.load(os.path.join(artifact_dir, "branch.pt"), map_location=device)

    # 4) assemble quantization (W4, and A-fake if requested)
    #    assemble_model usually modifies the struct/modules in place and/or returns the struct.
    model_struct = assemble_model(model_struct, ptq_config, branch, smooth, weights, use_fake_act)

    # 5) housekeeping
    del weights, smooth, branch; gc.collect(); torch.cuda.empty_cache()
    patched.eval().requires_grad_(False).to(device)
    return patched  # Quantized model instance (StatefulInfinity-compatible)
