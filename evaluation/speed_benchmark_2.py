import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- your compiled backend ----
import nunchaku._C as nunchaku_C
from nunchaku._C.ops import gemm_awq, gemv_awq

# ---- import the converter you provided ----
import sys
sys.path.append('../deepcompressor/')
import omniconfig

from deepcompressor.backend.nunchaku.convert import convert_to_nunchaku_w4x4y16_linear_state_dict
from nunchaku.models.text_encoders.linear import W4Linear


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16

def ceil_div(a, b): return (a + b - 1) // b

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
    def __init__(self, in_features, out_features,  lora_rank=0, use_fp4=False, device='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = torch.device(device)
        self.dtype = torch.float16

        if self.device.type == 'cuda':
            device_id = self.device.index
            if device_id is None:
                device_id = torch.cuda.current_device()
        else:
            # QuantizedGEMM is CUDA-only; keep 0 to satisfy the API but you'll need CUDA.
            device_id = 0

        self.backend = nunchaku_C.QuantizedGEMM()
        self.backend.init(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            use_fp4=use_fp4,
            bf16=(self.dtype == torch.bfloat16),
            deviceId=self.device.index if self.device.type == 'cuda' else 0,
            rank=lora_rank,   
        )

    def load_weights(self, packed_state):
        # Map scale keys to what the backend expects
        sd = dict(packed_state)  # copy
        # Backend typically looks for 'wscales'; the converter may return 'wcscales' or 'wtscale'
        if "wscales" not in sd:
            if "wcscales" in sd: sd["wscales"] = sd["wcscales"]
            elif "wtscale" in sd: sd["wscales"] = sd["wtscale"]
        # qweight must remain int8; everything else to fp16 on device
        proc = {}
        for k, v in sd.items():
            if k == "qweight":
                proc[k] = v.to(self.device).contiguous()
            else:
                proc[k] = v.to(self.device, self.dtype).contiguous()
        self.backend.loadDict(proc, False)

    def forward(self, x):
        x = x.to(self.device, self.dtype).contiguous()
        in_pad = ceil_div(self.in_features, 128) * 128
        if x.shape[-1] != in_pad:
            x = F.pad(x, (0, in_pad - self.in_features))
        y = self.backend.forward(x)
        return y[:, :self.out_features]
    

def make_modules(in_features, out_features, rank):
    # ----- Baseline FP16 -----
    W = torch.randn(out_features, in_features, device=device, dtype=dtype) * 0.05
    b = torch.randn(out_features, device=device, dtype=dtype) * 0.02
    baseline = nn.Linear(in_features, out_features, bias=True).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        baseline.weight.copy_(W); baseline.bias.copy_(b)

    # ----- Fake LoRA used by SVDQuant -----
    Ld = torch.randn(rank, in_features, device=device, dtype=dtype) * 0.2  # down [r, K]
    Lu = torch.randn(out_features, rank, device=device, dtype=dtype) * 0.2 # up   [N, r]

    # Per‑group scale (G=64) + smooth (all ones = no smoothing), good enough for timing
    scale  = torch.ones((out_features, 1, in_features//64, 1), device=device, dtype=dtype)
    smooth = torch.ones(in_features, device=device, dtype=dtype)

    # Pack once, reuse for both fused variants
    packed = convert_to_nunchaku_w4x4y16_linear_state_dict(
        weight=W, scale=scale, bias=b, smooth=smooth,
        lora=(Ld, Lu), smooth_fused=False, float_point=False, subscale=None
    )

    # ----- Old fused wrapper (your SVDQuantLinear) -----
    svdquant_layer = SVDQuantLinear(in_features, out_features, lora_rank=rank, use_fp4=False, device=device).eval()
    svdquant_layer.load_weights(packed)

    temp_linear = nn.Linear(in_features, out_features, bias=True, device=device, dtype=dtype)
    with torch.no_grad():
        temp_linear.weight.copy_(W)
        temp_linear.bias.copy_(b)

    # Instantiate your new AWQ layer
    # The W4Linear.from_linear handles the packing internally
    awq_layer = NunchakuLinearAWQ(temp_linear, group_size=64, lora_rank=rank)
    
    with torch.no_grad():
        awq_layer.lora_down.copy_(Ld.T)
        awq_layer.lora_up.copy_(Lu)
    return baseline, svdquant_layer, awq_layer


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
                  Ms=(1,2,4,8,16,32,64,128,256,512,1024,2048,4096), runs=50):

    baseline, old_fused, new_fused = make_modules(in_features, out_features, rank)

    print(f"\n== Benchmark: FP16 nn.Linear vs Fused (old) vs Fused (new) ==")
    print(f"device={device}, dtype={dtype}, in={in_features}, out={out_features}, rank={rank}")
    print("Note: fused paths pad K to 128 internally (baseline does not).")
    print(f"{'M':>6} | {'Linear (ms)':>12} | {'Old fused (ms)':>14} | {'New fused (ms)':>14} | {'Old spd↑':>8} | {'New spd↑':>8} | {'New/Old':>8}")
    print("-"*90)

    for M in Ms:
        x = (torch.randn(M, in_features, device=device, dtype=dtype) * 0.3 - 0.15)

        t_lin = bench(baseline,  x, runs=runs)
        t_old = bench(old_fused, x, runs=runs)
        t_new = bench(new_fused, x, runs=runs)

        sp_old = t_lin / max(t_old, 1e-9)
        sp_new = t_lin / max(t_new, 1e-9)
        rel    = t_old / max(t_new, 1e-9)

        print(f"{M:6d} | {t_lin:12.3f} | {t_old:14.3f} | {t_new:14.3f} | {sp_old:8.2f} | {sp_new:8.2f} | {rel:8.2f}")

if __name__ == "__main__":
    run_benchmark()