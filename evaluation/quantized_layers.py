import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
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
        use_fp4: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.bias = bool(bias)
        self.use_fp4 = bool(use_fp4)
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
        *,
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

import tempfile
from safetensors.torch import save_file

def _backend_load_adaptive(backend, state: dict, *, partial: bool = False):
    """
    Try in-memory loadDict first; otherwise save a temporary .safetensors to CPU and call load(path).
    """
    # 1) Preferred path: direct dict
    if hasattr(backend, "loadDict"):
        return backend.loadDict(state, bool(partial))

    # 2) Fallback: write to disk and use load(path, partial)
    if not hasattr(backend, "load"):
        raise AttributeError("QuantizedGEMM has neither loadDict nor load. Rebuild your extension.")

    # safetensors requires CPU tensors
    cpu_state = {}
    for k, v in state.items():
        if v is None:
            continue
        if hasattr(v, "device") and v.is_cuda:
            v = v.detach().to("cpu")
        cpu_state[k] = v.contiguous() if hasattr(v, "contiguous") else v

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        save_file(cpu_state, tmp_path)
        return backend.load(tmp_path, bool(partial))
    finally:
        # best effort cleanup
        import os
        try:
            os.remove(tmp_path)
        except OSError:
            pass

class SVDQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, lora_rank=0, bias = True, use_fp4=False, dtype = torch.bfloat16 , device="cuda"):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.device = torch.device(device)
        self.bias = bias
        self.dtype = dtype

        # resolve device id robustly
        if self.device.type == "cuda" and torch.cuda.is_available():
            device_id = self.device.index
            if device_id is None:
                device_id = torch.cuda.current_device()
        else:
            device_id = -1  # or 0 if your C++ expects >=0; both are common

        self.backend = nunchaku_C.QuantizedGEMM()

        bf16_flag = (self.dtype == torch.bfloat16)

        # Prefer the 7-arg signature if available; else fall back to 6-arg.
        try:
            # 7 args: (in, out, bias, use_fp4, bf16, deviceId, rank)
            self.backend.init(
                int(self.in_features),
                int(self.out_features),
                bias,
                bool(use_fp4),
                bool(bf16_flag),
                int(device_id),
                int(lora_rank),
            )
        except TypeError:
            # 6 args: (in, out, bias, use_fp4, bf16, deviceId)
            self.backend.init(
                int(self.in_features),
                int(self.out_features),
                bias,
                bool(use_fp4),
                bool(bf16_flag),
                int(device_id),
            )

    @torch.no_grad()
    def load_weights(self, packed_state: dict):
        # ensure tensors are contiguous (and typically GPU is fine for loadDict path)
        proc = {k: (v.contiguous() if isinstance(v, torch.Tensor) else v) for k, v in packed_state.items()}
        _backend_load_adaptive(self.backend, proc, partial=False)

    def forward(self, x):
        x = x.to(self.device, self.dtype).contiguous()
        #in_pad = ceil_div(self.in_features, 128) * 128
        #if x.shape[-1] != in_pad:
        #    x = F.pad(x, (0, in_pad - self.in_features))
        #y = self.backend.forward(x)
        #return y[:, :self.out_features]
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
                    bias=child.bias is not None, device=str(device)
                )
                W = child.weight.data.to(target_dtype)
                b = child.bias.data.to(target_dtype) if child.bias is not None else None

            # Initialize the new layer for the speed test
            Ld = torch.randn(lora_rank, in_features, device=device, dtype=target_dtype)
            Lu = torch.randn(out_features, lora_rank, device=device, dtype=target_dtype)
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

# -------- helpers for random packed weights (speed test only) -----------------

def _make_scale_4d(oc: int, ic: int, *, group_size: int, device, dtype) -> torch.Tensor:
    assert ic % group_size == 0, f"ic={ic} must be divisible by group_size={group_size}"
    ng = ic // group_size
    # per-group scales (fp32 is typical)
    return torch.ones((oc, 1, ng, 1), device=device, dtype=torch.float32)

def _lora_tuple(lora):
    if lora is None:
        return None
    if isinstance(lora, dict):
        return (lora["down"], lora["up"])
    if isinstance(lora, (list, tuple)):
        assert len(lora) >= 2
        return (lora[0], lora[1])
    raise TypeError("Unsupported LoRA container; expected dict or (down, up) tuple")

def _normalize_lora_keys(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # C++ side expects underscores in your current implementation
    return {k.replace("lora.down", "lora_down").replace("lora.up", "lora_up"): v for k, v in d.items()}

@torch.no_grad()
def _make_random_packed_fc(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    device: torch.device,
    dtype: torch.dtype,
    group_size: int = 64,
    lora_rank: int = 0,
) -> Dict[str, torch.Tensor]:
    W = torch.randn(out_features, in_features, device=device, dtype=dtype) * 0.002
    b = torch.randn(out_features, device=device, dtype=dtype) * 0.001 if bias else None
    scale = _make_scale_4d(out_features, in_features, group_size=group_size, device=device, dtype=dtype)*0+1
    smooth = torch.ones(in_features, device=device, dtype=dtype)

    lora = None
    if lora_rank and lora_rank > 0:
        lora = {
            "down": torch.randn(lora_rank, in_features, device=device, dtype=dtype) * 0.002,
            "up":   torch.randn(out_features, lora_rank, device=device, dtype=dtype) * 0.002,
        }

    packed = convert_to_nunchaku_w4x4y16_linear_state_dict(
        weight=W, scale=scale, bias=b, smooth=smooth,
        lora=_lora_tuple(lora), shift=None, smooth_fused=False, float_point=False, subscale=None,
    )
    return _normalize_lora_keys(packed)

# ------------- tiny wrapper that calls the fused module -----------------------

class _FusedMLPWrapper(nn.Module):
    """Drop-in module that mimics Linear->Act->Linear but delegates to SVDQuantFusedMLP."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        act_kind: str,               # "gelu" or "silu" (for now we only swap gelu reliably)
        bias: bool,
        use_fp4: bool,
        device: torch.device,
        dtype: torch.dtype,
        lora_rank: int,
        group_size: int,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.act_kind = act_kind
        self.bias = bias
        self.use_fp4 = use_fp4
        self.device = device
        self.dtype = dtype

        self.fused = SVDQuantFusedMLP(
            in_features=in_features,
            hidden_features=hidden_features,
            bias=bias,
            use_fp4=use_fp4,
            device=str(device),
            dtype=dtype,
        )

        # For speed tests: initialize with random packed weights
        packed_fc1 = _make_random_packed_fc(
            in_features=in_features, out_features=hidden_features,
            bias=bias, device=device, dtype=dtype, group_size=group_size, lora_rank=lora_rank,
        )
        packed_fc2 = _make_random_packed_fc(
            in_features=hidden_features, out_features=in_features,
            bias=bias, device=device, dtype=dtype, group_size=group_size, lora_rank=lora_rank,
        )
        self.fused.load_weights(packed_fc1=packed_fc1, packed_fc2=packed_fc2)

    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Activation is fused inside the backend; we keep act_kind for potential future routing.
        return self.fused(x)
    '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the backend sees what it expects
        if x.dtype is not self.dtype:
            x = x.to(self.dtype)
        #x = x.contiguous()

        y = self.fused(x)
        return y
    
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

                            device_i, dtype_i = _module_device_dtype(mod_i)
                            device_k, dtype_k = _module_device_dtype(mod_k)
                            device = device_i if device_i == device_k else device_k
                            dtype  = dtype_i if dtype_i == dtype_k else dtype_k
                            if dtype not in [torch.float16, torch.bfloat16]:
                                dtype = torch.float16

                            fused = _FusedMLPWrapper(
                                in_features=in_features,
                                hidden_features=hidden,
                                act_kind=act_kind,
                                bias=(mod_i.bias is not None) and (mod_k.bias is not None),
                                use_fp4=use_fp4,
                                device=device,
                                dtype=dtype,
                                lora_rank=lora_rank,
                                group_size=group_size,
                            )
                            replaced[path] = nn.Sequential(mod_i, mod_j, mod_k)  # keep originals
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
                        fused = _FusedMLPWrapper(
                            in_features=in_features,
                            hidden_features=hidden,
                            act_kind=act_kind,
                            bias=(fc1.bias is not None) and (fc2.bias is not None),
                            use_fp4=use_fp4,
                            device=device,
                            dtype=dtype,
                            lora_rank=lora_rank,
                            group_size=group_size,
                        )
                        # Replace the entire MLP submodule with the fused wrapper
                        # (actual replacement is done by the parent traversal below)
                        return {"__REPLACE_SELF__": fused}

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
            replaced[full_name] = child
            _replace_child(module, name, fused)

        # Merge child results
        replaced.update({k: v for k, v in result.items() if k != "__REPLACE_SELF__"})

    return replaced
