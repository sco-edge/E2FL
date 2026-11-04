# predictor/device_profiler.py

import torch
import torch.nn as nn
import time
import json
import os
from typing import Dict, Tuple, List

################################################################################
# Utility: FLOPs
################################################################################

def conv2d_flops(layer: nn.Conv2d, x_shape: Tuple[int, int, int, int]) -> int:
    N, Cin, H, W = x_shape
    Cout = layer.out_channels
    Kh, Kw = layer.kernel_size
    Sh, Sw = layer.stride
    Ph, Pw = layer.padding

    Hout = (H + 2*Ph - Kh) // Sh + 1
    Wout = (W + 2*Pw - Kw) // Sw + 1

    if layer.groups == 1:
        flops = N * Cout * Hout * Wout * (Cin * Kh * Kw)
    else:
        flops = N * Cin * Hout * Wout * (Kh * Kw)

    return int(flops)

def linear_flops(layer: nn.Linear, x_shape: Tuple[int, int]) -> int:
    N, in_f = x_shape
    return int(N * in_f * layer.out_features)

def eltwise_flops(shape):
    numel = 1
    for x in shape:
        numel *= x
    return numel

################################################################################
# DeviceProfiler
################################################################################

class DeviceProfiler:
    def __init__(self, device_name: str, save_dir="profile", mode="coarse"):
        """
        mode:
          - "coarse": op_type 기반 coarse β (PyTorch 연산 타입)
          - "fine": cuDNN convolution kernel algorithm 기반 β
        """
        self.mode = mode
        self.device_name = device_name

        # CUDA 여부 체크
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.device = torch.device("cuda")
            print(f"[Profiler] Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("[Profiler] CUDA unavailable → CPU mode (A-mode fallback only)")

        self.save_dir = os.path.join(save_dir, device_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.betas = {
            "mode": mode,
            "key_fwd": {},
            "key_bwd": {},
            "non_fwd": 0.0,
            "non_bwd": 0.0,
        }

    ############################################################################
    # Timing Utility
    ############################################################################
    def _measure(self, layer, x, iters=20):
        layer = layer.to(self.device)
        x = x.to(self.device)
        x.requires_grad_(True)

        # warm-up
        for _ in range(5):
            out = layer(x)
            loss = out.sum()
            loss.backward()
            layer.zero_grad(set_to_none=True)
            x.grad.zero_()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            # forward
            start.record()
            for _ in range(iters):
                out = layer(x)
            end.record()
            torch.cuda.synchronize()
            fwd_ms = start.elapsed_time(end) / iters

            # backward
            start.record()
            for _ in range(iters):
                out = layer(x)
                loss = out.sum()
                loss.backward()
                layer.zero_grad(set_to_none=True)
                x.grad.zero_()
            end.record()
            torch.cuda.synchronize()
            bwd_ms = start.elapsed_time(end) / iters
        else:
            # CPU fallback
            s = time.perf_counter()
            for _ in range(iters):
                out = layer(x)
            fwd_ms = (time.perf_counter() - s) * 1000 / iters

            s = time.perf_counter()
            for _ in range(iters):
                out = layer(x)
                loss = out.sum()
                loss.backward()
                layer.zero_grad(set_to_none=True)
                x.grad.zero_()
            bwd_ms = (time.perf_counter() - s) * 1000 / iters

        return fwd_ms, bwd_ms

    ############################################################################
    # Coarse-mode: PyTorch 연산 기반 β 프로파일링
    ############################################################################
    def _profile_coarse_mode(self):
        print("[Profiler] Running Coarse-mode (op_type) profiling")

        x = torch.randn(8, 32, 56, 56)

        # Conv3x3
        conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        flops = conv2d_flops(conv, (8, 32, 56, 56))
        fwd, bwd = self._measure(conv, x)
        self.betas["key_fwd"]["conv3x3"] = fwd / flops
        self.betas["key_bwd"]["conv3x3"] = bwd / flops

        # Conv1x1
        conv1 = nn.Conv2d(32, 64, kernel_size=1)
        flops = conv2d_flops(conv1, (8, 32, 56, 56))
        fwd, bwd = self._measure(conv1, x)
        self.betas["key_fwd"]["conv1x1"] = fwd / flops
        self.betas["key_bwd"]["conv1x1"] = bwd / flops

        # Depthwise
        dw = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        flops = conv2d_flops(dw, (8, 32, 56, 56))
        fwd, bwd = self._measure(dw, x)
        self.betas["key_fwd"]["depthwise"] = fwd / flops
        self.betas["key_bwd"]["depthwise"] = bwd / flops

        # Linear
        fc = nn.Linear(512, 512)
        x2 = torch.randn(32, 512)
        flops = linear_flops(fc, (32, 512))
        fwd, bwd = self._measure(fc, x2)
        self.betas["key_fwd"]["linear"] = fwd / flops
        self.betas["key_bwd"]["linear"] = bwd / flops

        # Non-key
        relu = nn.ReLU()
        x3 = torch.randn(8, 64, 56, 56)
        flops = eltwise_flops((8, 64, 56, 56))
        fwd, bwd = self._measure(relu, x3)
        self.betas["non_fwd"] = fwd / flops
        self.betas["non_bwd"] = bwd / flops

    ############################################################################
    # Fine-mode: cuDNN kernel algorithm profiling
    ############################################################################
    def _profile_fine_mode(self):
        # 여기서는 C++ 결과 JSON을 읽기만 함
        path = os.path.join(self.save_dir, "cudnn_betas_device.json")
        print(f"[Profiler] Loading Fine-mode betas from {path}")
        with open(path, "r") as f:
            data = json.load(f)

        # LATTE Training_Time_Estimator 형식에 맞게 그대로 삽입
        self.betas["key_fwd"] = data["key_fwd"]
        self.betas["key_bwd"] = data.get("key_bwd", {})
        self.betas["non_fwd"] = data.get("non_fwd", 0.0)
        self.betas["non_bwd"] = data.get("non_bwd", 0.0)


    ############################################################################
    # Public API
    ############################################################################
    def run(self):
        if self.mode == "coarse":
            self._profile_coarse_mode()
        elif self.mode == "fine":
            self._profile_fine_mode()
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def export(self, model_name):
        path = os.path.join(self.save_dir, f"{model_name}_betas.json")
        with open(path, "w") as f:
            json.dump(self.betas, f, indent=4)
        print(f"[Profiler] Saved → {path}")
