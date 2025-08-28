import os, gc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Iterable
from fnmatch import fnmatch


# ---- your compiled backend ----
import nunchaku._C as nunchaku_C
from nunchaku._C.ops import gemm_awq, gemv_awq

from deepcompressor.backend.nunchaku.convert import convert_to_nunchaku_w4x4y16_linear_state_dict
from nunchaku.models.text_encoders.linear import W4Linear


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True

def _dtype_is_bf16(dtype: torch.dtype) -> bool:
    return dtype in (torch.bfloat16,)

def _is_excluded(path: str, exclude_names) -> bool:
    return any(fnmatch(path, patt) or path.startswith(patt) for patt in exclude_names)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16

def ceil_div(a, b): return (a + b - 1) // b

class SVDQuantFusedMLP(nn.Module):
    """
    Python wrapper around nunchaku_C.QuantizedFusedMLP, which fuses:
        fc1 -> GELU (quant friendly) -> fc2
    The C++ expects a single state_dict with keys prefixed as 'fc1.' and 'fc2.'.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        bias: bool = True,
        act_kind: str,
        use_fp4: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.bias = bool(bias)
        self.use_fp4 = bool(use_fp4)
        self.act_kind = act_kind
        self.device = torch.device(device)
        self.dtype = dtype

        self.mod = nunchaku_C.QuantizedFusedMLP()
        # C++ init signature: (in_features, hidden_features, bias, use_fp4, bf16, deviceId)
        self.mod.init(
            int(in_features),
            int(hidden_features),
            bool(bias),
            bool(use_fp4),
            bool(_dtype_is_bf16(dtype)),
            int(torch.cuda.current_device() if self.device.type == "cuda" else -1),
        )

    @torch.no_grad()
    def load_weights(
        self,
        packed_fc1: Dict[str, torch.Tensor] = None,
        packed_fc2: Dict[str, torch.Tensor] = None,
        raw_fc1: Dict[str, torch.Tensor] = None,
        raw_fc2: Dict[str, torch.Tensor] = None,
        smooth_fused: bool = False,
        float_point: bool = False,
    ):
        """
        Load weights into the fused module.

        You can either:
          (A) pass 'packed_fc1' and 'packed_fc2': already in Nunchaku format
              with keys like: {'qweight','wscales','wcscales','wtscale','bias','smooth','lora.down','lora.up', ...}
          (B) pass 'raw_fc1' and 'raw_fc2': PyTorch-format tensors
              -> we will call convert_to_nunchaku_w4x4y16_linear_state_dict on each.

        The resulting keys are prefixed with 'fc1.' or 'fc2.' before calling C++ loadDict.
        """
        combined: Dict[str, torch.Tensor] = {}

        def _prefix(sd: Dict[str, torch.Tensor], pfx: str):
            for k, v in sd.items():
                if v is None:
                    continue
                combined[f"{pfx}{k}"] = v

        if packed_fc1 is not None and packed_fc2 is not None:
            _prefix(packed_fc1, "fc1.")
            _prefix(packed_fc2, "fc2.")
        elif raw_fc1 is not None and raw_fc2 is not None:
            # Expected raw dict keys (typical): 'weight', 'scale', optional: 'bias', 'smooth', 'lora', 'shift', 'subscale'
            def _to_packed(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                return convert_to_nunchaku_w4x4y16_linear_state_dict(
                    weight=d["weight"],
                    scale=d["scale"],
                    bias=d.get("bias"),
                    smooth=d.get("smooth"),
                    lora=d.get("lora"),
                    shift=d.get("shift"),
                    smooth_fused=smooth_fused,
                    float_point=float_point,
                    subscale=d.get("subscale"),
                )

            _prefix(_to_packed(raw_fc1), "fc1.")
            _prefix(_to_packed(raw_fc2), "fc2.")
        else:
            raise ValueError(
                "Provide either (packed_fc1 & packed_fc2) or (raw_fc1 & raw_fc2)."
            )

        # Ensure tensors live on current device and are contiguous for interop
        for k, v in list(combined.items()):
            if not isinstance(v, torch.Tensor):
                continue
            if v.device != self.device:
                v = v.to(self.device)
            combined[k] = v.contiguous()

        self.mod.loadDict(combined)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, *, in_features) → (B, *, in_features)
        # The C++ fused module handles fc1->gelu(quant)->fc2
        return self.mod.forward(x)
    
class GeneralPurposeW4Linear(W4Linear):
    """
    A modified W4Linear that is guaranteed to work for all batch sizes by
    exclusively using the robust gemv_awq kernel, processing larger batches
    in chunks. It requires a fixed group_size of 64.
    """
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        # Force group_size=64, which is required by the gemv_awq kernel
        super().__init__(
            in_features, 
            out_features, 
            bias=bias, 
            group_size=64, 
            **kwargs
        )

    @staticmethod
    def from_linear(linear: nn.Linear, **kwargs) -> "GeneralPurposeW4Linear":
        """
        Builds the layer from a standard nn.Linear layer, forcing group_size=64.
        """
        q_linear = W4Linear.from_linear(linear, **kwargs)
        # Create an instance of our new class and copy the packed weights
        new_layer = GeneralPurposeW4Linear(
            linear.in_features, 
            linear.out_features, 
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device
        )
        new_layer.qweight.copy_(q_linear.qweight)
        new_layer.scales.copy_(q_linear.scales)
        new_layer.scaled_zeros.copy_(q_linear.scaled_zeros)
        if linear.bias is not None:
            new_layer.bias.copy_(q_linear.bias)
        return new_layer

    @torch.no_grad()
    def forward(self, x):
        """
        Forward pass that uses a chunking strategy to rely only on the
        stable gemv_awq kernel.
        """
        x_reshaped = x.view(-1, x.shape[-1])
        M = x_reshaped.shape[0]

        # The gemv_awq kernel is templated to handle batches up to size 8
        chunk_size = 8
        
        y_chunks = []
        for i in range(0, M, chunk_size):
            x_chunk = x_reshaped[i : i + chunk_size]
            m_chunk = x_chunk.shape[0]
            
            y_chunk = gemv_awq(
                x_chunk,
                self.qweight,
                self.scales,
                self.scaled_zeros,
                m_chunk,
                self.out_features,
                self.in_features,
                self.group_size, # This will be 64
            )
            y_chunks.append(y_chunk)
            
        y = torch.cat(y_chunks, dim=0)
        
        # Reshape output to match the original input's shape dimensions
        y = y.view(*x.shape[:-1], self.out_features)

        if self.bias is not None:
            y = y + self.bias
            
        return y


class NunchakuLinearAWQ(nn.Module):
    """
    A W4A16 linear layer that uses the official Nunchaku W4Linear for its
    base computation and applies a LoRA path separately in PyTorch.
    """
    def __init__(self, linear_layer: nn.Linear, group_size: int, lora_rank: int = 0):
        super().__init__()
        
        # 1. Create the quantized W4A16 layer from the dense linear layer
        self.w4_layer = GeneralPurposeW4Linear.from_linear(linear_layer, group_size=group_size)
        
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.lora_rank = lora_rank
        
        # 2. Create LoRA parameters
        if self.lora_rank > 0:
            self.lora_down = nn.Parameter(
                torch.empty(self.in_features, lora_rank, device=linear_layer.weight.device, dtype=linear_layer.weight.dtype)
            )
            self.lora_up = nn.Parameter(
                torch.empty(self.out_features, lora_rank, device=linear_layer.weight.device, dtype=linear_layer.weight.dtype)
            )
            nn.init.normal_(self.lora_down, std=0.02)
            nn.init.normal_(self.lora_up, std=0.02)
        else:
            self.register_parameter('lora_down', None)
            self.register_parameter('lora_up', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Get the base result from the W4A16 layer
        y = self.w4_layer(x)
        
        # 2. Compute and add the LoRA path
        if self.lora_rank > 0:
            lora_result = (x @ self.lora_down) @ self.lora_up.T
            y = y + lora_result
            
        return y


class SVDQuantLinear(nn.Module):
    """
    Python wrapper for the Nunchaku W4A4+LoRA fused GEMM kernel.
    """
    def __init__(self, in_features, out_features, lora_rank, bias=True, use_fp4=False, dtype=torch.float16, device='cuda:0'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.bias_enabled = bias
        self.device = torch.device(device)
        self.dtype = dtype

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

        #output_padded = self.backend.forward(x)
        # Slice the output back to the original feature dimension
        #return output_padded[:, :self.out_features]
        x = x.to(self.device, self.dtype).contiguous()
        return self.backend.forward(x)
    


def swap_linear_for_svdquant(
    module: nn.Module,
    lora_rank: int = 32,
    exclude_names: List[str] = None,
    parent_name: str = ""
):
    """
    (Robust Version)
    Recursively traverses a module, replacing nn.Linear layers with SVDQuantLinear
    layers, while skipping any layers whose names are in the exclude_names list.
    """
    if exclude_names is None:
        exclude_names = []
        
    # Iterate over a copy of the children names to allow modification
    for name in list(module._modules.keys()):
        child = module._modules[name]
        full_name = f"{parent_name}.{name}" if parent_name else name
        
        # Check for exclusion before doing anything else
        if any(exc_name in full_name for exc_name in exclude_names):
            print(f"--> Skipping module and its children: {full_name}")
            continue

        if isinstance(child, nn.Linear):
            print(f"Swapping '{full_name}' with SVDQuantLinear...")
            
            in_features, out_features = child.in_features, child.out_features
            device = child.weight.device
            target_dtype = child.weight.dtype

            # Special handling for the SelfAttention.mat_qkv layer and its custom bias
            if name == 'mat_qkv' and "SelfAttention" in module.__class__.__name__:
                new_layer = SVDQuantLinear(
                    in_features, out_features, lora_rank=lora_rank,
                    bias=True, device=str(device), dtype=target_dtype
                )
                W = child.weight.data.to(target_dtype)
                b = torch.cat((module.q_bias, module.zero_k_bias, module.v_bias)).to(target_dtype)
            else:
                # Standard logic for all other nn.Linear layers
                new_layer = SVDQuantLinear(
                    in_features, out_features, lora_rank=lora_rank,
                    bias=child.bias is not None, device=str(device), dtype=target_dtype
                )
                W = child.weight.data.to(target_dtype)
                b = child.bias.data.to(target_dtype) if child.bias is not None else None

            # Initialize the new layer for the speed test
            Ld = torch.randn(lora_rank, in_features, device=device, dtype=target_dtype)*0.01
            Lu = torch.randn(out_features, lora_rank, device=device, dtype=target_dtype)*0.01
            scale = torch.ones((out_features, 1, in_features // 64, 1), device=device, dtype=target_dtype)
            smooth = torch.ones(in_features, device=device, dtype=target_dtype)
            packed_state = convert_to_nunchaku_w4x4y16_linear_state_dict(
                weight=W, scale=scale, bias=b, smooth=smooth, lora=(Ld, Lu)
            )
            new_layer.load_weights(packed_state)

            # Replace the old layer with the new one
            setattr(module, name, new_layer)

        # Always recurse into children, even after a swap (in case a swapped module has children)
        # This check prevents infinite recursion if a module has itself as a child
        if child is not module:
            swap_linear_for_svdquant(child, lora_rank, exclude_names, full_name)


# Reuse your fused wrapper from earlier in this file
# from deepcompressor.backend.nunchaku.quantized_layers import SVDQuantFusedMLP
# and the linear converter (we only use it to pack random weights for speed tests)
from deepcompressor.backend.nunchaku.convert import (
    convert_to_nunchaku_w4x4y16_linear_state_dict,
)


# ---------------------------- pattern utilities -------------------------------

def _is_linear(m: nn.Module) -> bool:
    return isinstance(m, nn.Linear)

def _is_act(m: nn.Module) -> Optional[str]:
    # returns "gelu" | "silu" | None
    if isinstance(m, nn.GELU):
        return "gelu"
    if isinstance(m, nn.SiLU):
        return "silu"
    return None

def _module_device_dtype(m: nn.Module) -> Tuple[torch.device, torch.dtype]:
    # Prefer first parameter/buffer; fallback to cuda:0 bf16
    for p in m.parameters(recurse=False):
        return p.device, p.dtype
    for b in m.buffers(recurse=False):
        return b.device, b.dtype
    # Fallbacks
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dev, torch.bfloat16

def _replace_child(parent: nn.Module, name: str, new: nn.Module):
    setattr(parent, name, new)

# ------------------------------ main entrypoint -------------------------------

@torch.no_grad()
def swap_mlp_for_svdquant_fused(
    module: nn.Module,
    *,
    lora_rank: int = 0,
    use_fp4: bool = False,
    only_gelu: bool = True,           # if True, skip SiLU for now (backend GELU is safest)
    group_size: int = 64,             # grouping for synthetic scales
    exclude_names: Optional[List[str]] = None,
    parent_name: str = "",
) -> Dict[str, nn.Module]:
    """
    Replace Linear->(GELU|SiLU)->Linear MLPs with SVDQuantFusedMLP, using RANDOM packed weights.
    Returns a dict path->original_module for optional rollback.

    v1 targets:
      - nn.Sequential(..., Linear, GELU/SILU, Linear, ...)
      - MLP-like submodules that expose .fc1/.act/.fc2 (all modules), replaced wholesale.
    """
    if exclude_names is None:
        exclude_names = []

    replaced: Dict[str, nn.Module] = {}

    # 1) Handle nn.Sequential by folding triplets
    if isinstance(module, nn.Sequential):
        # work on a copy of children list
        new_children = []
        i = 0
        children = list(module._modules.items())
        N = len(children)
        while i < N:
            name_i, mod_i = children[i]
            if i + 2 < N:
                name_j, mod_j = children[i + 1]
                name_k, mod_k = children[i + 2]
                act_kind = _is_act(mod_j)
                if _is_linear(mod_i) and act_kind is not None and _is_linear(mod_k):
                    if only_gelu and act_kind != "gelu":
                        # skip if we only fuse gelu
                        pass
                    else:
                        path = f"{parent_name}.{name_i}+{name_j}+{name_k}".strip(".")
                        if not _is_excluded(path, exclude_names):
                            # build fused wrapper
                            in_features = mod_i.in_features
                            hidden = mod_i.out_features
                            out_features = mod_k.out_features
                            if out_features != in_features:
                                # non-symmetric MLP (rare) — still supported by our fused wrapper (hidden→in),
                                # but for safety keep only symmetric C→hidden→C here
                                pass

                            device, dtype = _module_device_dtype(module)
                            if dtype not in (torch.float16, torch.bfloat16):
                                dtype = torch.bfloat16 

                            fused = SVDQuantFusedMLP(
                                in_features=in_features,
                                hidden_features=hidden,
                                act_kind=act_kind,          # "gelu" | "silu"
                                bias=(mod_i.bias is not None) and (mod_k.bias is not None),
                                use_fp4=use_fp4,
                                device=device,
                                dtype=dtype,
                            )
                            #replaced[path] = nn.Sequential(mod_i, mod_j, mod_k)  # keep originals
                            replaced[path] = "replaced"  # replace original
                            # Insert fused and skip the next two
                            new_children.append((name_i, fused))
                            i += 3
                            continue
            # default: keep original
            new_children.append((name_i, mod_i))
            i += 1

        # rebuild sequential if changed
        if len(new_children) != N:
            # 1) remove all current children
            for key in list(module._modules.keys()):
                module._modules.pop(key)
            # 2) re-add in the new order
            for name_new, mod_new in new_children:
                module.add_module(name_new, mod_new)

    # 2) Handle self-contained MLP-like submodules: has fc1/act/fc2 as Modules
    has_fc1 = hasattr(module, "fc1") and isinstance(getattr(module, "fc1"), nn.Module)
    has_fc2 = hasattr(module, "fc2") and isinstance(getattr(module, "fc2"), nn.Module)
    has_act = hasattr(module, "act") and isinstance(getattr(module, "act"), nn.Module)
    if has_fc1 and has_fc2 and has_act:
        fc1 = getattr(module, "fc1")
        act = getattr(module, "act")
        fc2 = getattr(module, "fc2")
        act_kind = _is_act(act)
        if _is_linear(fc1) and act_kind is not None and _is_linear(fc2):
            if not (only_gelu and act_kind != "gelu"):
                path = parent_name if parent_name else module.__class__.__name__
                if path not in exclude_names:
                    in_features = fc1.in_features
                    hidden = fc1.out_features
                    out_features = fc2.out_features
                    if out_features == in_features:
                        device, dtype = _module_device_dtype(module)
                        fused = SVDQuantFusedMLP(
                            in_features=in_features,
                            hidden_features=hidden,
                            act_kind=act_kind,          # "gelu" | "silu"
                            bias=(fc1.bias is not None) and (fc2.bias is not None),
                            use_fp4=use_fp4,
                            device=device,
                            dtype=dtype,
                        )
                        # Replace the entire MLP submodule with the fused wrapper
                        # (actual replacement is done by the parent traversal below)
                        return {"__REPLACE_SELF__": fused, path: "replaced"}

    # 3) Recurse over children, capturing replacements to apply at parent level
    for name, child in list(module.named_children()):
        full_name = f"{parent_name}.{name}".strip(".")
        if _is_excluded(full_name, exclude_names):
            continue

        result = swap_mlp_for_svdquant_fused(
            child,
            lora_rank=lora_rank,
            use_fp4=use_fp4,
            only_gelu=only_gelu,
            group_size=group_size,
            exclude_names=exclude_names,
            parent_name=full_name,
        )

        # If child asked to replace itself, perform the swap here
        if "__REPLACE_SELF__" in result:
            fused = result.pop("__REPLACE_SELF__")
            replaced[full_name] = "replaced"  
            _replace_child(module, name, fused)

        # Merge child results
        replaced.update({k: v for k, v in result.items() if k != "__REPLACE_SELF__"})

    return replaced


# --------------------------- artifact I/O -------------------------------------

def load_svdquant_artifacts(
    root: os.PathLike | str,
    *,
    map_location: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """
    Loads artifacts saved as:
      - model.pt   -> base weights & biases (fp16)
      - scale.pt   -> dequant scales per weight tensor
      - branch.pt  -> LoRA weights per layer: {'a.weight', 'b.weight'}
      - smooth.pt  -> SmoothQuant vector per layer
    Returns: (weights_sd, scale_sd, branch_sd, smooth_sd)
    """
    root = Path(root)
    weights_sd = torch.load(root / "model.pt", map_location=map_location)
    scale_sd   = torch.load(root / "scale.pt", map_location=map_location)
    branch_sd  = torch.load(root / "branch.pt", map_location=map_location)
    smooth_sd  = torch.load(root / "smooth.pt", map_location=map_location)
    return weights_sd, scale_sd, branch_sd, smooth_sd


# --------------------------- small helpers ------------------------------------

def _find_scale_key(k: str, scale_sd: Dict[str, torch.Tensor]) -> Optional[str]:
    """Finds the scale key for a weight tensor. Tries common variants."""
    cand = [f"{k}.scale.0", f"{k}.scale", f"{k}.scales.0", f"{k}.scales"]
    for c in cand:
        if c in scale_sd:
            return c
    # fallback: linear search for exact prefix match
    prefix = f"{k}.scale"
    for kk in scale_sd.keys():
        if kk.startswith(prefix):
            return kk
    return None


def _pack_linear_from_artifacts(
    path: str,
    weights_sd: Dict[str, torch.Tensor],
    scale_sd: Dict[str, torch.Tensor],
    branch_sd: Dict[str, Dict[str, torch.Tensor]],
    smooth_sd: Dict[str, torch.Tensor],
    *,
    target_dtype: torch.dtype,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Build packed state for a single linear, including fused SA QKV when `path` endswith '.sa.mat_qkv'.
    - weight/bias/smooth/LoRA are cast to target_dtype (fp16/bf16)
    - scale stays fp32
    - artifacts are kept on CPU
    """
    if target_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"target_dtype must be fp16/bf16, got {target_dtype}")

    is_qkv = path.split(".")[-1] == "mat_qkv"
    if path.split(".")[-1] == "mat_q":
        path = f"{'.'.join(path.split('.')[:-1])}.to_q"

    # ---------- helpers ----------
    def _scale_key_or_none(wk: str) -> Optional[str]:
        return _find_scale_key(wk, scale_sd)

    def _expand_scalar_scale_to_shaped(s: torch.Tensor, out_dim: int, in_dim: int, group_size: int = 64) -> torch.Tensor:
        # shaped expected by converter: [out, 1, ng, 1], ng = in_dim // group_size
        if s.dim() == 0:
            ng = (in_dim + group_size - 1) // group_size  # be safe if not divisible
            return s.float().repeat(out_dim, 1, ng, 1).contiguous()
        return s.float().contiguous()

    # ---------- fused SA QKV ----------
    if is_qkv:
        base = ".".join(path.split(".")[:-1])  # drop '.mat_qkv'
        q_path = f"{base}.to_q"
        k_path = f"{base}.to_k"
        v_path = f"{base}.to_v"

        wq_key, wk_key, wv_key = f"{q_path}.weight", f"{k_path}.weight", f"{v_path}.weight"
        bq_key, bk_key, bv_key = f"{q_path}.bias",   f"{k_path}.bias",   f"{v_path}.bias"

        # Require all three weights
        if not (wq_key in weights_sd and wk_key in weights_sd and wv_key in weights_sd):
            print(f"[svdquant-load] Missing one of Q/K/V weights for {path}")
            return None

        Wq = weights_sd[wq_key]  # [C, C]
        Wk = weights_sd[wk_key]
        Wv = weights_sd[wv_key]
        # Concatenate along out (row) dimension: [3C, C]
        W = torch.cat([Wq, Wk, Wv], dim=0).to(target_dtype).contiguous()

        # Bias: concatenate; use zeros for any missing part; if all missing, set None
        C_out = Wq.shape[0]
        bq = weights_sd.get(bq_key, None)
        bk = weights_sd.get(bk_key, None)
        bv = weights_sd.get(bv_key, None)

        parts = []
        any_bias = False
        for b_part in (bq, bk, bv):
            if b_part is None:
                parts.append(torch.zeros(C_out, dtype=target_dtype))
            else:
                parts.append(b_part.to(target_dtype).contiguous())
                any_bias = True
        b = torch.cat(parts, dim=0) if any_bias else None

        # Scales: concat along out if shaped; expand scalars to shaped first
        sq_key = _scale_key_or_none(wq_key)
        sk_key = _scale_key_or_none(wk_key)
        sv_key = _scale_key_or_none(wv_key)
        if not (sq_key and sk_key and sv_key):
            print(f"[svdquant-load] Missing one of Q/K/V scales for {path}")
            return None

        Sq = _expand_scalar_scale_to_shaped(scale_sd[sq_key], Wq.shape[0], Wq.shape[1])
        Sk = _expand_scalar_scale_to_shaped(scale_sd[sk_key], Wk.shape[0], Wk.shape[1])
        Sv = _expand_scalar_scale_to_shaped(scale_sd[sv_key], Wv.shape[0], Wv.shape[1])
        S = torch.cat([Sq, Sk, Sv], dim=0).contiguous()  # keep fp32

        # Smooth: per-input vector; pick from to_q (you said it's stored there)
        sm_key = q_path
        if sm_key not in smooth_sd:
            print(f"[svdquant-load] MISSING smooth for {path} (looked under {sm_key})")
            return None
        smooth = smooth_sd[sm_key].to(target_dtype).contiguous()

        # LoRA: you said it's already fused under to_q; load if present
        lora = None
        entry = branch_sd.get(q_path)
        if isinstance(entry, dict) and "a.weight" in entry and "b.weight" in entry:
            Ld = entry["a.weight"].to(target_dtype).contiguous()  # (r, C)
            Lu = entry["b.weight"].to(target_dtype).contiguous()  # (3C, r)
            # normalize orientation if needed
            if Ld.dim() == 2 and Ld.shape[0] != min(Ld.shape):
                Ld = Ld.t().contiguous()
            if Lu.dim() == 2 and Lu.shape[0] < Lu.shape[1]:
                Lu = Lu.t().contiguous()
            lora = (Ld, Lu)

        packed = convert_to_nunchaku_w4x4y16_linear_state_dict(
            weight=W, scale=S, bias=b, smooth=smooth,
            lora=lora, shift=None, smooth_fused=False, float_point=False, subscale=None,
        )
        return packed

    # ---------- normal (non-QKV) linear ----------
    w_key = f"{path}.weight"
    b_key = f"{path}.bias"
    s_key = _find_scale_key(w_key, scale_sd)
    sm_key = path

    if w_key not in weights_sd:
        print(f"[svdquant-load] MISSING weight for {path}")
        return None
    if s_key is None:
        print(f"[svdquant-load] MISSING scale for {path} (looked under {w_key}.*)")
        return None
    if sm_key not in smooth_sd:
        print(f"[svdquant-load] MISSING smooth for {path}")
        return None

    W = weights_sd[w_key].to(target_dtype).contiguous()
    b = weights_sd.get(b_key, None)
    if b is not None:
        b = b.to(target_dtype).contiguous()

    S = _expand_scalar_scale_to_shaped(scale_sd[s_key], W.shape[0], W.shape[1])  # keep fp32
    smooth = smooth_sd[sm_key].to(target_dtype).contiguous()

    # LoRA (non-fused) if present
    lora = None
    entry = branch_sd.get(sm_key)
    if isinstance(entry, dict) and "a.weight" in entry and "b.weight" in entry:
        Ld = entry["a.weight"].to(target_dtype).contiguous()
        Lu = entry["b.weight"].to(target_dtype).contiguous()
        if Ld.dim() == 2 and Ld.shape[0] != min(Ld.shape):  # (in, r) -> (r, in)
            Ld = Ld.t().contiguous()
        if Lu.dim() == 2 and Lu.shape[0] < Lu.shape[1]:      # (r, out) -> (out, r)
            Lu = Lu.t().contiguous()
        lora = (Ld, Lu)

    packed = convert_to_nunchaku_w4x4y16_linear_state_dict(
        weight=W, scale=S, bias=b, smooth=smooth,
        lora=lora, shift=None, smooth_fused=False, float_point=False, subscale=None,
    )
    return packed


# --------------------------- main loader (linear only) ------------------------

@torch.no_grad()
def load_svdquant_weights(
    model: nn.Module,
    artifacts: Tuple[
        Dict[str, torch.Tensor],                      # weights_sd (model.pt)
        Dict[str, torch.Tensor],                      # scale_sd   (scale.pt)
        Dict[str, Dict[str, torch.Tensor]],           # branch_sd  (branch.pt)
        Dict[str, torch.Tensor],                      # smooth_sd  (smooth.pt)
    ],
    *,
    strict: bool = False,
    dry_run: bool = False,
    only_paths: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    weights_sd, scale_sd, branch_sd, smooth_sd = artifacts
    report: Dict[str, str] = {}
    only_paths = set(only_paths) if only_paths is not None else None

    def _target_dtype_of(mod: nn.Module) -> torch.dtype:
        dt = getattr(mod, "dtype", None)
        if dt is not None:
            return dt
        # prefer bf16 over fp32 if ambiguous
        pd = next((p.dtype for p in mod.parameters(recurse=False)), None)
        bd = next((b.dtype for b in mod.buffers(recurse=False)), None)
        guess = pd or bd or torch.bfloat16
        return torch.bfloat16 if guess in (torch.float32, torch.bfloat16) else guess

    for path, mod in model.named_modules():
        if only_paths is not None and path not in only_paths:
            continue

        # --- fused MLPs ---
        if isinstance(mod, (SVDQuantFusedMLP)):
            td = _target_dtype_of(mod)

            p1 = _pack_linear_from_artifacts(f"{path}.fc1", weights_sd, scale_sd, branch_sd, smooth_sd, target_dtype=td)
            p2 = _pack_linear_from_artifacts(f"{path}.fc2", weights_sd, scale_sd, branch_sd, smooth_sd, target_dtype=td)

            if p1 is None or p2 is None:
                report[path] = "missing-artifacts"
                msg = f"[svdquant-load-fused] {path}: " + ("fc1 missing " if p1 is None else "") + ("fc2 missing" if p2 is None else "")
                print(msg)
                if strict:
                    raise KeyError(msg)
                continue

            if dry_run:
                report[path] = "would-load"
                print(f"[svdquant-load-fused] {path}: OK (dry-run)")
            else:
                if hasattr(mod, "load_weights"):
                    mod.load_weights(p1, p2)
                elif hasattr(mod, "mod") and hasattr(mod.mod, "load_weights"):
                    mod.mod.load_weights(p1, p2)
                else:
                    raise AttributeError(f"[svdquant-load-fused] {path}: no load_weights(p1,p2) available")
                report[path] = "loaded"

        # --- single linears ---
        elif isinstance(mod, SVDQuantLinear):
            td = _target_dtype_of(mod)
            packed = _pack_linear_from_artifacts(path, weights_sd, scale_sd, branch_sd, smooth_sd, target_dtype=td)

            if packed is None:
                report[path] = "missing-artifacts"
                if strict:
                    raise KeyError(f"[svdquant-load] missing-artifacts: {path}")
                continue

            if dry_run:
                report[path] = "would-load"
                print(f"[svdquant-load] {path}: OK (dry-run)")
            else:
                if hasattr(mod, "load_weights"):
                    mod.load_weights(packed)
                elif hasattr(mod, "backend") and hasattr(mod.backend, "loadDict"):
                    mod.backend.loadDict(packed, False)
                else:
                    raise AttributeError(f"[svdquant-load] {path}: no load method found")
                report[path] = "loaded"

    return report