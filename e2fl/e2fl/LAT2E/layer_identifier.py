# predictor/layer_identifier.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn


# ------------------------------------------------------------
# FLOPs helpers (A/B 공용)
# ------------------------------------------------------------

def conv2d_flops(layer: nn.Conv2d, x_shape: Tuple[int, int, int, int]) -> int:
    """Conv2d FLOPs 계산 (forward 기준)."""
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
        # depthwise 등 groups > 1
        flops = N * Cin * Hout * Wout * (Kh * Kw)

    return int(flops)


def linear_flops(layer: nn.Linear, x_shape: Tuple[int, int]) -> int:
    """Linear FLOPs."""
    N, in_features = x_shape
    return int(N * in_features * layer.out_features)


def eltwise_flops(shape: Tuple[int, ...]) -> int:
    """Element-wise 연산 FLOPs 근사."""
    numel = 1
    for v in shape:
        numel *= v
    return int(numel)


# ------------------------------------------------------------
# Shape extractor (hook 기반)
# ------------------------------------------------------------

def extract_layer_shapes(
    model: nn.Module,
    input_size: Tuple[int, int] = (32, 32),
    in_channels: int = 3,
    device: str = "cpu",
) -> Dict[str, Dict[str, torch.Size]]:
    """
    model.named_modules() 에 대해 leaf module 들의
    input / output shape 를 hook 으로 수집.
    """
    model = model.to(device)
    model.eval()

    shape_dict: Dict[str, Dict[str, torch.Size]] = {}
    hooks = []

    for name, module in model.named_modules():
        # leaf module 만 대상으로
        if len(list(module.children())) > 0:
            continue

        def make_hook(layer_name: str):
            def hook(mod, inp, out):
                x_in = inp[0]
                x_out = out
                if isinstance(x_out, (tuple, list)):
                    x_out = x_out[0]
                shape_dict[layer_name] = {
                    "in": x_in.shape,
                    "out": x_out.shape,
                }
            return hook

        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

    dummy = torch.randn(1, in_channels, input_size[0], input_size[1], device=device)
    with torch.no_grad():
        _ = model(dummy)

    for h in hooks:
        h.remove()

    return shape_dict


# ------------------------------------------------------------
# Dataclass for per-layer info
# ------------------------------------------------------------

@dataclass
class LayerInfo:
    name: str
    kind: str  # "key" or "nonkey"
    algo: Optional[str]  # key layer면 algo 이름, 아니면 None
    flops: int
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]


# ------------------------------------------------------------
# LayerIdentifier: A/B mode 겸용
# ------------------------------------------------------------

class LayerIdentifier:
    """
    A/B-mode layer identifier

    - mode="coarse": coarse 모드
        Conv2d → conv3x3 / conv1x1 / depthwise
        Linear → linear
        나머지 → non-key (C_non에만 반영)

    - mode="fine": fine 모드
        algo_overrides[name] 가 있으면 그걸 우선 사용
        없으면 A-mode 규칙으로 fallback
    """

    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, int] = (32, 32),
        in_channels: int = 3,
        device: Optional[str] = None,
        mode: str = "coarse",
        algo_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        assert mode in ("coarse", "fine"), f"Unsupported mode: {mode}"
        self.model = model
        self.input_size = input_size
        self.in_channels = in_channels
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mode = mode
        self.algo_overrides = algo_overrides or {}

    # --------------------
    # 알고리즘 선택 (A-mode 규칙)
    # --------------------
    @staticmethod
    def _infer_algo_A(
        module: nn.Module,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
    ) -> Tuple[bool, Optional[str], int]:
        """
        A-mode 기준으로 key/nonkey, algo_name, flops 계산.
        algo_name 은 device_profiler 가 만든 betas 의 key 와 맞춰야 함:
          - "conv3x3", "conv1x1", "depthwise", "linear"
        """
        # Conv2d
        if isinstance(module, nn.Conv2d):
            if len(in_shape) != 4:
                return False, None, 0
            N, Cin, H, W = in_shape
            Kh, Kw = module.kernel_size

            # algo 이름 결정
            if module.groups == Cin == module.out_channels:
                algo = "depthwise"
            elif Kh == 1 and Kw == 1:
                algo = "conv1x1"
            else:
                # 3x3 뿐 아니라 5x5 등도 여기로 몰아넣되,
                # FLOPs 는 실제 kernel size 로 계산 → beta(conv3x3) * FLOPs
                algo = "conv3x3"

            flops = conv2d_flops(module, (N, Cin, H, W))
            return True, algo, flops

        # Linear
        if isinstance(module, nn.Linear):
            if len(in_shape) != 2:
                # (N, C, H, W) 로 들어왔을 수도 있음 → flatten 된 걸로 가정
                if len(in_shape) == 4:
                    N, C, H, W = in_shape
                    in_shape_lin = (N, C * H * W)
                else:
                    return False, None, 0
            else:
                in_shape_lin = in_shape

            algo = "linear"
            flops = linear_flops(module, in_shape_lin)
            return True, algo, flops

        # Non-key
        # out_shape 기준으로 eltwise 플롭 근사
        flops = eltwise_flops(out_shape)
        return False, None, flops

    # --------------------
    # B-mode: override + A-mode fallback
    # --------------------
    def _infer_algo_B(
        self,
        name: str,
        module: nn.Module,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
    ) -> Tuple[bool, Optional[str], int]:
        """
        B-mode:
          - algo_overrides[name] 가 있으면 그 algo 사용
          - 없으면 A-mode 규칙으로 추론
        """
        # 먼저 A-mode 로 기본값 계산
        is_key, algo_A, flops = self._infer_algo_A(module, in_shape, out_shape)

        # override 있으면 교체
        if name in self.algo_overrides:
            algo_override = self.algo_overrides[name]
            if algo_override is None:
                # 강제로 non-key 처리 하고 싶은 경우
                return False, None, flops
            return True, algo_override, flops

        return is_key, algo_A, flops

    # --------------------
    # 메인 식별 루프
    # --------------------
    def identify(self) -> Dict:
        """
        전체 모델에 대해:
          - per-layer info
          - algo_selection, C_key, C_non
        를 계산해서 dict 로 반환.
        """
        print(f"[LayerIdentifier] mode={self.mode}, device={self.device}")

        shapes = extract_layer_shapes(
            self.model,
            input_size=self.input_size,
            in_channels=self.in_channels,
            device=str(self.device),
        )

        layer_infos: List[LayerInfo] = []
        algo_selection: List[str] = []
        C_key: List[float] = []
        C_non: float = 0.0

        for name, module in self.model.named_modules():
            # leaf module만
            if len(list(module.children())) > 0:
                continue
            if name not in shapes:
                # hook 에 안 걸린 레이어 (예: register_forward_pre_hook 등) 은 스킵
                continue

            in_shape = tuple(shapes[name]["in"])
            out_shape = tuple(shapes[name]["out"])

            if self.mode == "coarse":
                is_key, algo, flops = self._infer_algo_A(module, in_shape, out_shape)
            else:
                is_key, algo, flops = self._infer_algo_B(name, module, in_shape, out_shape)

            if is_key and algo is not None:
                kind = "key"
                algo_selection.append(algo)
                C_key.append(float(flops))
            else:
                kind = "nonkey"
                C_non += float(flops)

            layer_infos.append(
                LayerInfo(
                    name=name,
                    kind=kind,
                    algo=algo,
                    flops=int(flops),
                    in_shape=in_shape,
                    out_shape=out_shape,
                )
            )

        result = {
            "mode": self.mode,
            "algo_selection": algo_selection,  # Training_Time_Estimator.algo_selection
            "C_key": C_key,                    # Training_Time_Estimator.C_key
            "C_non": C_non,                    # Training_Time_Estimator.C_non
            "layers_order": [li.name for li in layer_infos],
            "layers": {
                li.name: {
                    "kind": li.kind,
                    "algo": li.algo,
                    "flops": li.flops,
                    "in_shape": list(li.in_shape),
                    "out_shape": list(li.out_shape),
                }
                for li in layer_infos
            },
        }
        return result

    # --------------------
    # JSON 저장
    # --------------------
    @staticmethod
    def save_json(data: Dict, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[LayerIdentifier] Saved algo profile → {path}")


# ------------------------------------------------------------
# CLI 예시 (원하면 수정해서 사용)
# ------------------------------------------------------------
'''
if __name__ == "__main__":
    import argparse
    from torchvision import models

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mobilenet_v2")
    parser.add_argument("--mode", type=str, default="coarse", choices=["coarse", "fine"])
    parser.add_argument("--out", type=str, default="profile/mobilenet_v2_algo.json")
    parser.add_argument("--input_h", type=int, default=32)
    parser.add_argument("--input_w", type=int, default=32)
    parser.add_argument("--in_channels", type=int, default=3)
    args = parser.parse_args()

    # 필요한 모델만 간단히 매핑
    if args.model == "mobilenet_v2":
        net = models.mobilenet_v2(weights=None)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    identifier = LayerIdentifier(
        net,
        input_size=(args.input_h, args.input_w),
        in_channels=args.in_channels,
        mode=args.mode,
        algo_overrides=None,  # B-mode 쓸 때 dict로 넣으면 됨
    )
    info = identifier.identify()
    LayerIdentifier.save_json(info, args.out)
'''