# predictor/device_profiler.py

import torch
import torch.nn as nn
import time
import json
import os
from typing import Dict, Tuple


########################################
# FLOPs 계산 함수
########################################

def conv2d_flops(layer: nn.Conv2d, x_shape: Tuple[int, int, int, int]) -> int:
    """Conv2d FLOPs 계산 (forward pass 기준)."""
    N, Cin, H, W = x_shape
    Cout = layer.out_channels
    Kh, Kw = layer.kernel_size
    Sh, Sw = layer.stride
    Ph, Pw = layer.padding

    Hout = (H + 2 * Ph - Kh) // Sh + 1
    Wout = (W + 2 * Pw - Kw) // Sw + 1

    if layer.groups == 1:
        flops = N * Cout * Hout * Wout * (Cin * Kh * Kw)
    else:  
        flops = N * Cin * Hout * Wout * (Kh * Kw)

    return int(flops)


def linear_flops(layer: nn.Linear, x_shape: Tuple[int, int]) -> int:
    N, features = x_shape
    return int(N * features * layer.out_features)


def eltwise_flops(x_shape: Tuple[int, ...]) -> int:
    numel = 1
    for v in x_shape:
        numel *= v
    return int(numel)


########################################
# DeviceProfiler 본체
########################################

class DeviceProfiler:

    def __init__(self, device_name: str, save_dir: str = "profile"):
        # GPU 여부 로깅
        print(f"[Profiler] Init: device_name = {device_name}")
        print(f"[Profiler] torch.cuda.is_available = {torch.cuda.is_available()}")
        print(f"[Profiler] device_count = {torch.cuda.device_count()}")

        if torch.cuda.is_available():
            print(f"[Profiler] Using CUDA device: {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda")
        else:
            print("[Profiler] WARNING: CUDA unavailable → using CPU")
            self.device = torch.device("cpu")

        self.device_name = device_name
        self.save_dir = os.path.join(save_dir, device_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.betas = {
            "key_fwd": {},
            "key_bwd": {},
            "non_fwd": 0.0,
            "non_bwd": 0.0
        }

    ########################################
    # forward / backward 시간 측정
    ########################################
    def _measure_fwd_bwd(self, layer: nn.Module, x: torch.Tensor, iters: int = 20):
        print(f"[Profiler] Measuring layer = {layer.__class__.__name__}")

        layer = layer.to(self.device)
        x = x.to(self.device)
        x.requires_grad_(True)

        # Warm-up
        print("[Profiler] Warm-up start")
        for _ in range(5):
            out = layer(x)
            loss = out.sum()
            try:
                loss.backward()
            except Exception as e:
                print("[ERROR] backward failed during warm-up:", e)
                return 0.0, 0.0
            layer.zero_grad(set_to_none=True)
            x.grad.zero_()
        print("[Profiler] Warm-up done")

        # GPU 모드
        if self.device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()

            # Forward timing
            print("[Profiler] Forward timing start")
            starter.record()
            for _ in range(iters):
                out = layer(x)
            ender.record()
            torch.cuda.synchronize()
            fwd_ms = starter.elapsed_time(ender) / iters
            print(f"[Profiler] Forward avg ms = {fwd_ms}")

            # Backward timing
            print("[Profiler] Backward timing start")
            starter.record()
            for _ in range(iters):
                out = layer(x)
                loss = out.sum()
                loss.backward()
                layer.zero_grad(set_to_none=True)
                x.grad.zero_()
            ender.record()
            torch.cuda.synchronize()
            bwd_ms = starter.elapsed_time(ender) / iters
            print(f"[Profiler] Backward avg ms = {bwd_ms}")

            return fwd_ms, bwd_ms

        # CPU 모드
        print("[Profiler] CPU timing mode")
        start = time.perf_counter()
        for _ in range(iters):
            out = layer(x)
        fwd_ms = (time.perf_counter() - start) * 1000 / iters
        print(f"[Profiler] Forward avg ms = {fwd_ms}")

        start = time.perf_counter()
        for _ in range(iters):
            out = layer(x)
            loss = out.sum()
            loss.backward()
            layer.zero_grad(set_to_none=True)
            x.grad.zero_()
        bwd_ms = (time.perf_counter() - start) * 1000 / iters
        print(f"[Profiler] Backward avg ms = {bwd_ms}")

        return fwd_ms, bwd_ms


    ########################################
    # Conv profiling
    ########################################
    def profile_conv(self):
        x = torch.randn(8, 32, 56, 56)

        # Conv3x3
        conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        flops3 = conv2d_flops(conv3, (8, 32, 56, 56))
        fwd, bwd = self._measure_fwd_bwd(conv3, x)
        self.betas["key_fwd"]["conv3x3"] = fwd / flops3
        self.betas["key_bwd"]["conv3x3"] = bwd / flops3

        # Conv1x1
        conv1 = nn.Conv2d(32, 64, kernel_size=1)
        flops1 = conv2d_flops(conv1, (8, 32, 56, 56))
        fwd, bwd = self._measure_fwd_bwd(conv1, x)
        self.betas["key_fwd"]["conv1x1"] = fwd / flops1
        self.betas["key_bwd"]["conv1x1"] = bwd / flops1

        # Depthwise
        dw = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        flops_dw = conv2d_flops(dw, (8, 32, 56, 56))
        fwd, bwd = self._measure_fwd_bwd(dw, x)
        self.betas["key_fwd"]["depthwise"] = fwd / flops_dw
        self.betas["key_bwd"]["depthwise"] = bwd / flops_dw

    ########################################
    # Linear profiling
    ########################################
    def profile_linear(self):
        layer = nn.Linear(512, 512)
        x = torch.randn(32, 512)
        flops = linear_flops(layer, (32, 512))
        fwd, bwd = self._measure_fwd_bwd(layer, x)
        self.betas["key_fwd"]["linear"] = fwd / flops
        self.betas["key_bwd"]["linear"] = bwd / flops

    ########################################
    # Non-key profiling
    ########################################
    def profile_nonkey(self):
        relu = nn.ReLU()
        x = torch.randn(8, 64, 56, 56)
        flops = eltwise_flops((8, 64, 56, 56))
        fwd, bwd = self._measure_fwd_bwd(relu, x)
        self.betas["non_fwd"] = fwd / flops
        self.betas["non_bwd"] = bwd / flops

    ########################################
    # 저장
    ########################################
    def export(self, model_name: str):
        path = os.path.join(self.save_dir, f"{model_name}_betas.json")
        with open(path, "w") as f:
            json.dump(self.betas, f, indent=4)
        print(f"[Profiler] Saved beta file → {path}")
