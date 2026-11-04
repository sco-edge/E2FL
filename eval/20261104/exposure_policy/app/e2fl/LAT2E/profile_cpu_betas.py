import torch
import torch.nn as nn
import time
import json
import os
import argparse
from collections import defaultdict

torch.set_num_threads(4)      # Pi5 성능 위해 thread 수 조절 가능
device = "cpu"


# -----------------------------------------------------------
# UTILS
# -----------------------------------------------------------
def measure_time(func, iters=30, warmup=5):
    # warmup
    for _ in range(warmup):
        func()

    start = time.time()
    for _ in range(iters):
        func()
    end = time.time()

    return (end - start) * 1000.0 / iters  # ms


def compute_conv_flops(N, C_in, H, W, C_out, Kh, Kw, groups, Hout, Wout):
    return float(N) * Hout * Wout * C_out * (C_in // groups) * Kh * Kw


# -----------------------------------------------------------
# MAIN PROFILING LOGIC
# -----------------------------------------------------------
def profile_convs():
    conv_configs = [
        # conv3x3
        (8, 32, 56, 56, 64, 3, 3, 1),
        # depthwise
        (8, 64, 28, 28, 64, 3, 3, 64),
        # conv1x1
        (8, 128, 28, 28, 128, 1, 1, 1),
    ]

    key_fwd = {}
    key_bwd = {}

    for (N, Cin, H, W, Cout, Kh, Kw, groups) in conv_configs:
        x = torch.randn(N, Cin, H, W, device=device)
        conv = nn.Conv2d(Cin, Cout, kernel_size=(Kh, Kw),
                         padding=Kh//2, groups=groups).to(device)

        # output shape
        with torch.no_grad():
            y = conv(x)
        Hout, Wout = y.shape[2], y.shape[3]

        flops = compute_conv_flops(N, Cin, H, W, Cout, Kh, Kw, groups, Hout, Wout)

        # layer type
        if Kh == 1 and Kw == 1:
            layer_type = "conv1x1"
        elif groups == Cin:
            layer_type = "depthwise"
        else:
            layer_type = "conv3x3"

        # forward ----------------------------------------------------
        def fwd(): conv(x)

        t_ms = measure_time(fwd)
        key = f"{layer_type}_fwd"
        key_fwd[key] = t_ms / flops

        # backward ---------------------------------------------------
        x_req = x.clone().requires_grad_(True)
        y = conv(x_req)
        loss = y.sum()

        def bwd():
            loss = conv(x_req).sum()
            loss.backward(retain_graph=True)

        t_ms_bwd = measure_time(bwd)
        key = f"{layer_type}_bwd"
        key_bwd[key] = t_ms_bwd / flops

    return key_fwd, key_bwd


def profile_nonconv():
    N = 1
    C = 256
    H = 64
    W = 64
    x = torch.randn(N, C, H, W, device=device, requires_grad=True)

    nc_flops = float(N * C * H * W)

    non_fwd = {}
    non_bwd = {}

    # ReLU ----------------------------------------------------------
    relu = nn.ReLU().to(device)

    def relu_fwd(): relu(x)

    t_ms = measure_time(relu_fwd)
    non_fwd["relu"] = t_ms / nc_flops

    def relu_bwd():
        out = relu(x)
        loss = out.sum()
        loss.backward(retain_graph=True)

    t_ms = measure_time(relu_bwd)
    non_bwd["relu"] = t_ms / nc_flops

    # Add -----------------------------------------------------------
    x2 = torch.randn_like(x)

    def add_fwd(): x + x2

    t_ms = measure_time(add_fwd)
    non_fwd["add"] = t_ms / nc_flops

    # BN ------------------------------------------------------------
    bn = nn.BatchNorm2d(C).to(device)

    def bn_fwd(): bn(x)

    t_ms = measure_time(bn_fwd)
    non_fwd["bn"] = t_ms / nc_flops

    def bn_bwd():
        out = bn(x)
        loss = out.sum()
        loss.backward(retain_graph=True)

    t_ms = measure_time(bn_bwd)
    non_bwd["bn"] = t_ms / nc_flops

    return non_fwd, non_bwd


# -----------------------------------------------------------
# JSON SAVE LOGIC
# -----------------------------------------------------------
def save_json(device_name, model_name, key_fwd, key_bwd, non_fwd, non_bwd):
    home = os.path.expanduser("~")
    outdir = f"{home}/EEFL/E2FL/predictor/profile/{device_name}"
    os.makedirs(outdir, exist_ok=True)

    # fine.json (LATTE style)
    fine = {
        "mode": "fine",
        "key_fwd": key_fwd,
        "key_bwd": key_bwd,
        "non_fwd": sum(non_fwd.values()) / len(non_fwd),
        "non_bwd": sum(non_bwd.values()) / len(non_bwd),
    }

    with open(f"{outdir}/{model_name}_betas_fine.json", "w") as f:
        json.dump(fine, f, indent=4)

    # fine_full.json (layer-type breakdown)
    fine_full = {
        "mode": "fine_full",
        "key_fwd": key_fwd,
        "key_bwd": key_bwd,
        "non_fwd": non_fwd,
        "non_bwd": non_bwd,
    }

    with open(f"{outdir}/{model_name}_betas_fine_full.json", "w") as f:
        json.dump(fine_full, f, indent=4)

    print("[Saved]")
    print(f" - {outdir}/{model_name}_betas_fine.json")
    print(f" - {outdir}/{model_name}_betas_fine_full.json")


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("device_name", type=str)
    parser.add_argument("model_name", type=str)
    args = parser.parse_args()

    print("Profiling convolution layers...")
    key_fwd, key_bwd = profile_convs()

    print("Profiling non-convolution layers...")
    non_fwd, non_bwd = profile_nonconv()

    print("Saving...")
    save_json(args.device_name, args.model_name, key_fwd, key_bwd, non_fwd, non_bwd)


if __name__ == "__main__":
    main()

#python3 profile_cpu_betas.py RPi5 mobilenet_v2
