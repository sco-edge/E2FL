# predictor/run_profiler.py

import argparse
import os
import torchvision.models as models

from layer_identifier import LayerIdentifier
from device_profiler import DeviceProfiler


SUPPORTED_MODELS = {
    "mobilenet_v2": models.mobilenet_v2,
    # 필요하면 여기 계속 추가
    # "resnet18": models.resnet18,
    # "efficientnet_b0": models.efficientnet_b0,
}


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_profile_root() -> str:
    """
    모든 프로파일 결과의 루트 디렉토리.
    예: ~/EEFL/E2FL/predictor/profile
    """
    home = os.path.expanduser("~")
    root = os.path.join(home, "EEFL", "E2FL", "predictor", "profile")
    ensure_dir(root)
    return root


# ---------------------------------------------------------
# 1) LayerIdentifier 실행 (coarse/fine 모드)
# ---------------------------------------------------------
def run_layer_identifier(model_name: str, mode: str, device_for_hooks: str) -> str:
    print(f"[1] LayerIdentifier: model={model_name}, mode={mode}, device={device_for_hooks}")

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    model_ctor = SUPPORTED_MODELS[model_name]
    model = model_ctor(weights=None)

    li = LayerIdentifier(
        model=model,
        input_size=(224, 224),
        in_channels=3,
        device=device_for_hooks,
        mode=mode,  # "coarse" or "fine"
    )

    result = li.identify()

    profile_root = get_profile_root()
    # 파일명에 mode를 붙여서 coarse/fine 둘 다 저장 가능하게
    out_path = os.path.join(profile_root, f"{model_name}_workload_{mode}.json")
    LayerIdentifier.save_json(result, out_path)

    print(f"[run_profiler] Saved workload → {out_path}")
    print(f"  - #key layers : {len(result['algo_selection'])}")
    print(f"  - C_non       : {result['C_non']:.3e}")
    return out_path


# ---------------------------------------------------------
# 2) DeviceProfiler 실행 (coarse/fine 모드: op_type / kernel algo)
# ---------------------------------------------------------
def run_device_profiler(model_name: str, device_name: str, mode: str) -> str:
    """
    mode:
      - "coarse": op_type 기반 (네가 지금 구현해둔 coarse β)
      - "fine": cuDNN kernel algorithm 기반 (추후 제대로 다듬어서 쓸 예정)
    """
    print(f"[2] DeviceProfiler: device={device_name}, mode={mode}")

    profile_root = get_profile_root()

    profiler = DeviceProfiler(
        device_name=device_name,
        save_dir=profile_root,  # profile_root/<device_name>/... 구조로 저장
        mode=mode,
    )

    profiler.run()
    betas_path = profiler.export(model_name)

    print(f"[run_profiler] Saved betas → {betas_path}")
    return betas_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="e.g., mobilenet_v2")
    parser.add_argument("--device", required=True, help="logical device name (RPi5, jetson_orin_nx, ...)")
    parser.add_argument(
        "--mode",
        choices=["coarse", "fine"],
        default="coarse",
        help="LayerIdentifier mode and DeviceProfiler mode: coarse=op_type, fine=kernel algo (default: coarse)",
    )
    args = parser.parse_args()

    # hook용 device 선택 (LayerIdentifier용)
    try:
        import torch
        if args.device == "jetson_orin_nx" and torch.cuda.is_available():
            hook_device = "cuda"
        else:
            hook_device = "cpu"
    except ImportError:
        hook_device = "cpu"

    print("========================================")
    print("  LATTE-style Profiler")
    print("  model     :", args.model)
    print("  device    :", args.device)
    print("  mode   :", args.mode)
    print("========================================")

    workload_path = run_layer_identifier(
        model_name=args.model,
        mode=args.mode,
        device_for_hooks=hook_device,
    )

    betas_path = run_device_profiler(
        model_name=args.model,
        device_name=args.device,
        mode=args.mode,
    )

    print("========================================")
    print("  DONE")
    print("  workload json :", workload_path)
    print("  betas json    :", betas_path)
    print("========================================")


if __name__ == "__main__":
    main()
