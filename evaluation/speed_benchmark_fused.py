import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.quantized_layers import SVDQuantFusedMLP
from deepcompressor.backend.nunchaku.convert import (
    convert_to_nunchaku_w4x4y16_linear_state_dict,
)

import importlib
C = importlib.import_module("nunchaku._C")
print("ext:", C.__file__)
print("QuantizedFusedMLP:", hasattr(C, "QuantizedFusedMLP"))
m = C.QuantizedFusedMLP()  # should construct

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class BaselineMLP(nn.Module):
    """FP16/BF16 baseline: Linear -> GELU -> Linear (matches fused block topology)."""
    def __init__(self, in_features: int, hidden_features: int, bias: bool = True, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device, dtype=dtype)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))

def make_scale_4d(oc: int, ic: int, *, group_size: int, device, dtype):
    """Return per-group scale shaped [oc, 1, ng, 1], ng = ic // group_size."""
    assert ic % group_size == 0, f"ic={ic} must be divisible by group_size={group_size}"
    ng = ic // group_size
    return torch.ones((oc, 1, ng, 1), device=device, dtype=dtype)

def make_random_linear_raw(
    in_features: int,
    out_features: int,
    *,
    device: str = "cuda",
    dtype = torch.float16,
    bias: bool = True,
    group_size: int = 64,      # <<< use 64 (divides typical dims: 1024, 1536, 2048, 2816, 4096, 8192)
    with_lora: bool = False,
    lora_rank: int = 0,
) -> dict:
    """
    Fabricate a raw linear 'state' in torch format compatible with convert_to_nunchaku_w4x4y16_linear_state_dict.
    """
    weight = torch.randn(out_features, in_features, device=device, dtype=dtype) * 0.02
    scale  = make_scale_4d(out_features, in_features, group_size=group_size,
                           device=device, dtype=dtype)  # converter expects fp16 scales
    bias_t = torch.randn(out_features, device=device, dtype=dtype) * 0.01 if bias else None
    smooth = torch.ones(in_features, device=device, dtype=dtype)  # 1D, len == ic

    raw = {"weight": weight, "scale": scale, "bias": bias_t, "smooth": smooth}

    if with_lora and lora_rank > 0:
        raw["lora"] = {
            "down": torch.randn(lora_rank, in_features, device=device, dtype=dtype) * 0.02,
            "up":   torch.randn(out_features, lora_rank, device=device, dtype=dtype) * 0.02,
            "scale": torch.tensor(1.0, device=device, dtype=dtype),
        }
    return raw

def _normalize_lora_keys(d: dict) -> dict:
    return {k.replace("lora.down", "lora_down").replace("lora.up", "lora_up"): v for k, v in d.items()}

@torch.no_grad()
def build_packed_for_fused_mlp(
    in_features: int,
    hidden_features: int,
    *,
    device: str = "cuda",
    bias: bool = True,
    with_lora: bool = False,
    lora_rank: int = 0,
) -> Tuple[dict, dict]:
    """
    Returns (packed_fc1, packed_fc2) for the fused MLP.
    packed dicts are suitable for SVDQuantFusedMLP.load_weights(packed_fc1=..., packed_fc2=...).
    """
    raw_fc1 = make_random_linear_raw(
        in_features=in_features,
        out_features=hidden_features,
        device=device,
        bias=bias,
        with_lora=with_lora,
        lora_rank=lora_rank,
    )
    raw_fc2 = make_random_linear_raw(
        in_features=hidden_features,
        out_features=in_features,
        device=device,
        bias=bias,
        with_lora=with_lora,
        lora_rank=lora_rank,
    )

    packed_fc1 = convert_to_nunchaku_w4x4y16_linear_state_dict(
        weight=raw_fc1["weight"],
        scale=raw_fc1["scale"],
        bias=raw_fc1.get("bias"),
        smooth=raw_fc1.get("smooth"),
        lora=_lora_tuple(raw_fc1.get("lora")),   # <<< HERE
        shift=None,
        smooth_fused=False,
        float_point=False,
        subscale=None,
    )

    packed_fc2 = convert_to_nunchaku_w4x4y16_linear_state_dict(
        weight=raw_fc2["weight"],
        scale=raw_fc2["scale"],
        bias=raw_fc2.get("bias"),
        smooth=raw_fc2.get("smooth"),
        lora=_lora_tuple(raw_fc2.get("lora")),   # <<< HERE
        shift=None,
        smooth_fused=False,
        float_point=False,
        subscale=None,
    )

    # Optional but safer with your C++ loader using "lora_down"/"lora_up"
    packed_fc1 = _normalize_lora_keys(packed_fc1)
    packed_fc2 = _normalize_lora_keys(packed_fc2)
    return packed_fc1, packed_fc2

def bench(model, x, runs=50, warmup=20) -> float:
    # Sync for stable timing
    if x.is_cuda:
        torch.cuda.synchronize()
    for _ in range(warmup):
        y = model(x)
    if x.is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        y = model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if torch.sum(torch.isinf(y)) > 0:
            print('There are Inf values!!')
        if torch.sum(torch.isnan(y)) > 0:
            print('There are nan values!!')
        times.append((t1 - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]  # median ms

def _lora_tuple(lora):
    """Normalize lora input to a (down, up) tuple for the converter."""
    if lora is None:
        return None
    if isinstance(lora, dict):
        return (lora["down"], lora["up"])
    if isinstance(lora, (list, tuple)):
        assert len(lora) >= 2, "LoRA must provide (down, up)"
        return (lora[0], lora[1])
    raise TypeError("Unsupported LoRA container; expected dict or (down, up) tuple")

def run_benchmark(
    in_features: int,
    hidden_features: int,
    *,
    batch: int = 16,
    seq: int = 1024,
    bias: bool = True,
    dtype = torch.bfloat16,
    device: str = "cuda",
    runs: int = 50,
    warmup: int = 20,
    with_lora: bool = True,
    lora_rank: int = 32,
    use_fp4: bool = False,
):
    print(f"\n== Fused MLP bench: in={in_features}, hidden={hidden_features}, "
          f"B={batch}, T={seq}, bias={bias}, dtype={dtype}, lora={with_lora}:{lora_rank}, fp4={use_fp4}")

    # Inputs
    x = torch.randn(batch, seq, in_features, device=device, dtype=dtype)

    # Baseline FP16/BF16
    baseline = BaselineMLP(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
    t_baseline = bench(baseline, x, runs=runs, warmup=warmup)
    print(f"Baseline (Linear->GELU->Linear) median: {t_baseline:.2f} ms")

    # Fused quantized
    fused = SVDQuantFusedMLP(
        in_features=in_features,
        hidden_features=hidden_features,
        bias=bias,
        use_fp4=use_fp4,
        device=device,
        dtype=dtype,
    )
    packed_fc1, packed_fc2 = build_packed_for_fused_mlp(
        in_features=in_features,
        hidden_features=hidden_features,
        device=device,
        bias=bias,
        with_lora=with_lora,
        lora_rank=lora_rank,
    )
    fused.load_weights(packed_fc1=packed_fc1, packed_fc2=packed_fc2)

    t_fused = bench(fused, x, runs=runs, warmup=warmup)
    print(f"Fused W4A4 (QuantizedFusedMLP) median: {t_fused:.2f} ms")
    print(f"Speedup vs baseline: {t_baseline / t_fused:.2f}×")

if __name__ == "__main__":
    torch.manual_seed(0)

    cfgs = [
        (2048, 8192, 1),
        (2048, 8192, 16),
        (2048, 8192, 64),
        (2048, 8192, 256),
        (8192, 2048, 256),
        (2048, 8192, 1024),
        (2048, 8192, 2048),
        (2048, 8192, 48*48),
        (2048, 8192, 4096),
    ]
    for in_f, hid_f, num_tokens in cfgs:
        run_benchmark(
            in_features=in_f,
            hidden_features=hid_f,
            batch=2,
            seq=num_tokens,
            runs=50,
            warmup=10,
            with_lora=True,
            lora_rank=32,
            dtype=torch.bfloat16,
            device="cuda",
            use_fp4=False,
        )
