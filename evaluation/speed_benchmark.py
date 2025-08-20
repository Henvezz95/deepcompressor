

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- your compiled backend ----
import nunchaku._C as nunchaku_C

# ---- import the converter you provided ----
import sys
sys.path.append('../deepcompressor/')
import omniconfig

from deepcompressor.backend.nunchaku.convert import convert_to_nunchaku_w4x4y16_linear_state_dict

print([n for n in dir(nunchaku_C.ops) if 'awq' in n.lower()])
print([n for n in dir(nunchaku_C) if 'gem' in n.lower()])

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16

def ceil_div(a, b): return (a + b - 1) // b

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
    def __init__(self, in_features, out_features, lora_rank=0, use_fp4=False, device="cuda"):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.device = torch.device(device)
        self.dtype = torch.float16  # set to bfloat16 if you want the BF16 path

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
                True,
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
                True,
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
        in_pad = ceil_div(self.in_features, 128) * 128
        if x.shape[-1] != in_pad:
            x = F.pad(x, (0, in_pad - self.in_features))
        y = self.backend.forward(x)
        return y[:, :self.out_features]

def make_modules(in_features, out_features, rank):
    # ----- Baseline: plain FP16 Linear (original network behavior) -----
    W = torch.randn(out_features, in_features, device=device, dtype=dtype) * 0.05
    b = torch.randn(out_features, device=device, dtype=dtype) * 0.02

    baseline = nn.Linear(in_features, out_features, bias=True).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        baseline.weight.copy_(W)
        baseline.bias.copy_(b)

    # ----- Fused kernel: W4A4 + (optional) LoRA -----
    # LoRA branch that SVDQuant adds. Keep it non-zero because that’s what you’ll deploy.
    Ld = torch.randn(rank, in_features, device=device, dtype=dtype) * 0.2  # lora_down
    Lu = torch.randn(out_features, rank, device=device, dtype=dtype) * 0.2 # lora_up

    # Scales/smooth for the converter:
    # - If you don’t have calibrated per-group scales yet, per-tensor scale=1.0 is fine for benchmarking kernel cost.
    # - Smooth=ones means no smoothing effect (again fine for timing).
    scale = torch.ones((out_features, 1, in_features//64, 1), device=device, dtype=dtype) 
    smooth = torch.ones(in_features, device=device, dtype=dtype)

    packed = convert_to_nunchaku_w4x4y16_linear_state_dict(
        weight=W,
        scale=scale,
        bias=b,
        smooth=smooth,
        lora=(Ld, Lu),   # API expects (down, up); down is [rank, in]
        smooth_fused=False,
        float_point=False,
        subscale=None,
    )

    fused = SVDQuantLinear(in_features, out_features, lora_rank=rank, use_fp4=False, device=device).eval()
    fused.load_weights(packed)
    return baseline, fused

def bench(model, x, runs=50, warmup=20):
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            starter.record()
            _ = model(x)
            ender.record()
            ender.synchronize()
            times.append(starter.elapsed_time(ender))  
    times.sort()
    return times[len(times)//2]

def run_benchmark(in_features=2048, out_features=8192, rank=32,
                  Ms=(1,2,4,8,16,32,64,128,256,512,1024,2048,4096), runs=100):
    baseline, fused = make_modules(in_features, out_features, rank)

    print(f"\n== Benchmark: FP16 nn.Linear vs W4A4(+LoRA) fused ==")
    print(f"device={device}, dtype={dtype}, in={in_features}, out={out_features}, rank={rank}")
    print("Note: fused path pads K to multiple of 128 internally (baseline does not).")
    print(f"{'M':>6} | {'Linear FP16 (ms)':>17} | {'Fused W4A4 (ms)':>16} | {'Speedup':>8}")
    print("-"*56)

    for M in Ms:
        x = (torch.randn(M, in_features, device=device, dtype=dtype) * 0.3 - 0.15)
        #time.sleep(0.1)
        t_lin = bench(baseline, x, runs=runs) 
        #time.sleep(0.1)
        t_fus = bench(fused, x, runs=runs) 
        sp = t_lin / max(t_fus, 1e-9)
        print(f"{M:6d} | {t_lin:17.3f} | {t_fus:16.3f} | {sp:8.2f}x")

if __name__ == "__main__":
    run_benchmark(in_features=2048, out_features=2048)
    run_benchmark(in_features=4096, out_features=4096)
    run_benchmark(in_features=2048, out_features=8192)
    run_benchmark(in_features=8192, out_features=2048)