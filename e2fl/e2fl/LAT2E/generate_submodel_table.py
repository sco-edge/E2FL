import os
import json
import argparse
import torch
import torchvision.models as models

from latte_selector import LatteSelector
from latte_submodel import (
    get_conv_output_channels,
    generate_width_submodels,
    save_submodel_table,
)
from e2fl.LAT2E.Training_Time_Estimator import Training_Time_Estimator


# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
home_dir = os.path.expanduser("~")
PROFILE_DIR = f"{home_dir}/EEFL/E2FL/predictor/profile/"


# -------------------------------------------------------------
# Helper: Load PyTorch model by name
# -------------------------------------------------------------
def load_model(model_name: str):
    """Load a PyTorch vision model by name."""
    model_name = model_name.lower()

    if model_name == "mobilenet_v2":
        return models.mobilenet_v2(weights=None)
    elif model_name == "resnet18":
        return models.resnet18(weights=None)
    elif model_name == "resnet50":
        return models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


# -------------------------------------------------------------
# Helper: compute FLOPs for windows
# -------------------------------------------------------------
def compute_flops_for_submodel(model, windows):
    """
    Compute FLOPs for a submodel defined by channel windows.
    For now we approximate:
      scaled_flops = original_flops * (used_channels / total_channels)
    """

    total_flops = 0.0
    total_nonkey = 0.0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            out_ch = module.out_channels
            k_h, k_w = module.kernel_size
            stride_h, stride_w = module.stride
            groups = module.groups

            # Original FLOPs (per LATTE coarse definition)
            # 2 * Cout * H_out * W_out * Cin * K * K / groups
            Cin = module.in_channels
            Cout = out_ch
            flops_full = 2 * Cin * Cout * k_h * k_w / groups

            # Scale using used channels
            if name in windows:
                used = windows[name][0][1]
                scale = used / Cout
            else:
                scale = 1.0

            total_flops += flops_full * scale

        # Non-conv layers (BN/ReLU/etc)
        # LATTE approximates these into C_non as constant factor
        elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.ReLU,
                                 torch.nn.ReLU6, torch.nn.GELU)):
            total_nonkey += 1e5  # simple small constant placeholder

    return total_flops, total_nonkey


# -------------------------------------------------------------
# Main generation function
# -------------------------------------------------------------
def create_submodel_table(
    model_name: str,
    selector_path: str,
    betas_path: str,
    output_path: str,
    num_submodels: int = 200,
):
    print(f"[LATTE] Generating submodel table for model: {model_name}")
    print(f"[LATTE] Output path: {output_path}")

    # 1. Load model
    model = load_model(model_name)
    model.eval()

    # 2. Load selector
    print("[LATTE] Loading MLP selector...")
    selector = LatteSelector(selector_path, device="cpu")

    # 3. Load betas
    print("[LATTE] Loading beta profile...")
    with open(betas_path, "r") as f:
        betas = json.load(f)

    # 4. Load conv channels
    print("[LATTE] Collecting conv layer channels...")
    conv_channels = get_conv_output_channels(model)

    # 5. Generate scales
    print("[LATTE] Generating width multipliers...")
    scales = list(torch.linspace(1.0, 0.3, num_submodels).tolist())

    # 6. Generate SubModelConfig list
    print("[LATTE] Creating raw submodels...")
    raw_submodels = generate_width_submodels(conv_channels, scales)

    # 7. Initialize Training Time Estimator
    tte = Training_Time_Estimator()
    tte.load_profiled_betas(betas)

    # 8. Populate each submodel with detailed metadata
    print("[LATTE] Populating submodel metadata (algo_selection, FLOPs, T_single)...")

    for sub in raw_submodels:
        # Extract windows
        windows = sub.windows

        # FLOPs
        C_key, C_non = compute_flops_for_submodel(model, windows)

        # Algorithm selection for this submodel
        # We use base model shapes + selector
        algo_map, feats, names = selector.build_algo_selection(model)

        # T_single
        # We approximate algo_selection as list aligned with conv order
        algo_list = [algo_map[n] for n in names if n in algo_map]

        try:
            T_single = tte.estimate_single_pass(algo_list, [C_key], C_non)
            sub.T_single = float(T_single)
        except Exception as e:
            print(f"[WARN] Failed T_single estimation for submodel {sub.id}: {e}")
            sub.T_single = None

        sub.mem_est = C_key / 1e6  # cheap normalization

    # 9. Save as JSON
    print("[LATTE] Saving submodels...")
    save_submodel_table(model_name, raw_submodels, output_path)

    print("[LATTE] DONE!")


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name: mobilenet_v2/resnet18/resnet50")

    parser.add_argument("--selector", type=str, required=True,
                        help="Path to ML.pt")

    parser.add_argument("--betas", type=str, required=True,
                        help="Path to betas_fine.json")

    parser.add_argument("--num", type=int, default=200,
                        help="Number of submodels to generate")

    args = parser.parse_args()

    output = os.path.join(PROFILE_DIR, f"{args.model}_submodel_table.json")

    create_submodel_table(
        args.model,
        args.selector,
        args.betas,
        output,
        num_submodels=args.num,
    )

'''
python generate_submodel_table.py \
    --model mobilenet_v2 \
    --selector ~/EEFL/E2FL/predictor/profile/ML.pt \
    --betas ~/EEFL/E2FL/predictor/profile/mobilenet_v2_betas_fine.json \
    --num 200
'''