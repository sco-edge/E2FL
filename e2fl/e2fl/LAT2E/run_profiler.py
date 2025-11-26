# predictor/run_profiler.py
import argparse
import os
import json
import torchvision.models as models

from layer_identifier import LayerIdentifier
from device_profiler import DeviceProfiler

from Layer_Selector import (
    GenericLayerSelector,
    extract_layer_features_mobilenet_v2,
    build_algo_selection,
    save_algo_selection_json,
)

SUPPORTED_MODELS = {
    "mobilenet_v2": models.mobilenet_v2,
    # "resnet18": models.resnet18,
    # "efficientnet_b0": models.efficientnet_b0,
    # "resnet50": models.resnet50,
}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def run_layer_identifier(model_name: str):
    print(f"[1] Running LayerIdentifier for {model_name}...")

    model = SUPPORTED_MODELS[model_name](weights=None)
    li = LayerIdentifier(
        model=model,
        input_shape=(1, 3, 224, 224),
        model_name=model_name,
    )
    layer_info = li.analyze()

    home_dir = os.path.expanduser("~")
    path = f"{home_dir}/EEFL/E2FL/predictor/profile"
    ensure_dir(path)

    out_path = f"{path}/{model_name}_workload.json"
    with open(out_path, "w") as f:
        json.dump(layer_info, f, indent=2)

    print(f"[+] Saved workload profile → {out_path}")
    return layer_info


def run_device_profiler(model_name, device_name):
    print(f"[2] Running device profiler for {device_name}...")

    profiler = DeviceProfiler(device_name=device_name)
    profiler.profile_conv()
    profiler.profile_linear()
    profiler.profile_nonkey()
    profiler.export(model_name)


def run_algo_selector(model_name, device="cpu"):
    print(f"[3] Running Layer_Selector for {model_name}...")

    model = SUPPORTED_MODELS[model_name](weights=None).to(device)

    # 1) Extract features
    features, names = extract_layer_features_mobilenet_v2(
        model,
        input_size=(32, 32),
        in_channels=3,
        device=device,
    )

    print(f"[+] Extracted {len(features)} layer features")

    if len(features) == 0:
        print("[!] No features extracted, skipping algo_selection.")
        return None

    # 2) Build selector
    input_dim = len(features[0])
    selector = GenericLayerSelector(input_dim=input_dim, num_algos=3).to(device)

    # 3) Build algo_selection result
    algo_selection = build_algo_selection(
        layer_names=names,
        feature_vectors=features,
        selector=selector,
        device=device,
        include_features=True,  # JSON에도 feature 포함
    )

    # 4) Save
    home_dir = os.path.expanduser("~")
    save_path = f"{home_dir}/EEFL/E2FL/predictor/profile/{model_name}_algo_selection.json"
    save_algo_selection_json(algo_selection, save_path)

    print(f"[+] Saved algo_selection → {save_path}")
    return algo_selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    if args.device == "jetson_orin_nx":
        import torch
        print("=== CUDA CHECK START ===")
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("Current device:", torch.cuda.current_device())
            print("Device name:", torch.cuda.get_device_name(0))
            a = torch.randn(1000, 1000).cuda()
            b = torch.randn(1000, 1000).cuda()
            torch.cuda.synchronize()
            print("GPU matmul OK:", torch.matmul(a, b).sum().item())
        print("=== CUDA CHECK END ===")
        selector_device = "cuda"
    else:
        selector_device = "cpu"

    print("===================================")
    print(" LATTE Predictor Pipeline (A+B+C)")
    print("===================================")

    run_layer_identifier(args.model)
    run_device_profiler(args.model, args.device)
    run_algo_selector(args.model, device=selector_device)

    print("===================================")
    print("    Completed ALL profiling")
    print("===================================")


if __name__ == "__main__":
    main()


# python3 run_profiler.py --model mobilenet_v2 --device jetson_orin_nx
# python3 run_profiler.py --model mobilenet_v2 --device RPi5
