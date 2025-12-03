"""
Layer_Selector.py

Generic layer-wise algorithm selector for LATTE-style training-time prediction.

- Extracts per-layer features from a given model (starting with Conv2d layers).
- Feeds features into a small neural network (GenericLayerSelector) which
  predicts an "algorithm class" for each layer (e.g., 0/1/2).
- Produces an `algo_selection` mapping:
      { layer_name: algo_id, ... }
  which can be stored in your model_profile JSON and used by Training_Time_Estimator.

This file is written to work out-of-the-box with MobileNetV2, but the
feature extraction and selector are generic enough to be reused for other
architectures (ResNet, EfficientNet, etc.) by just changing the feature extractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Generic Selector Network
# ---------------------------------------------------------------------------

HOOK_LAYER_TYPES = (
    nn.Conv2d,
    nn.Linear,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.ReLU6,
    nn.GELU,
    nn.MaxPool2d,
    nn.AvgPool2d,
)

class GenericLayerSelector(nn.Module):
    """
    A generic neural network to select an algorithm class for each layer.

    Input:
        - Feature vector of length `input_dim` describing a layer.
          (e.g., op_type, input/output shapes, kernel size, stride, groups, etc.)

    Output:
        - Logits of shape (N, num_algos), where each row corresponds to one layer,
          and argmax over dim=1 gives the algorithm class ID.
    """

    def __init__(
        self,
        input_dim: int,
        num_algos: int = 3,
        use_feature_importance: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_algos = num_algos

        # Attention-like linear transform
        self.attention = nn.Linear(input_dim, input_dim)

        # Feature importance vector (trainable)
        self.feature_importance = nn.Parameter(torch.ones(input_dim))

        # MLP backbone: input_dim -> 256 -> 128 -> 64 -> 32 -> 16 -> num_algos
        hidden1 = max(64, min(256, input_dim * 4))  # a bit adaptive but bounded
        hidden2 = max(32, hidden1 // 2)
        hidden3 = max(16, hidden2 // 2)

        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, num_algos)

        if use_feature_importance:
            self.initialize_feature_importance()
        else:
            nn.init.ones_(self.feature_importance)

    def initialize_feature_importance(self) -> None:
        """
        Initialize feature_importance similar to original LATTE code.

        Assumes feature ordering:
            0: op_type
            1: input_c
            2: input_h
            3: input_w
            4: output_c
            5: output_h
            6: output_w
            7: kernel_h
            8: kernel_w
            9: stride_h
            10: stride_w
            11: groups
            12: has_bias
        If input_dim < 11, this will only partially apply.
        """
        with torch.no_grad():
            self.feature_importance.fill_(1.0)

            # Only apply if indices exist
            if self.input_dim > 9:
                self.feature_importance.data[9] = 2.0  # stride_height
            if self.input_dim > 10:
                self.feature_importance.data[10] = 2.0  # stride_width
            if self.input_dim > 2:
                self.feature_importance.data[2] = 0.5  # input_h
            if self.input_dim > 3:
                self.feature_importance.data[3] = 0.5  # input_w
            if self.input_dim > 4:
                self.feature_importance.data[4] = 0.5  # output_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (N, input_dim) or (input_dim,), float32
        returns logits: (N, num_algos)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.size(1) == self.input_dim, (
            f"Expected feature dim {self.input_dim}, got {x.size(1)}"
        )

        # Attention-style weighting
        attention_weights = torch.sigmoid(self.attention(x))  # (N, input_dim)
        combined_weights = attention_weights * self.feature_importance  # broadcast

        x = x * combined_weights

        # MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)  # logits

        return x

    @torch.no_grad()
    def predict_algorithms(self, features: torch.Tensor) -> torch.Tensor:
        """
        Convenience method:
            features: (N, input_dim)
        returns:
            algo_ids: (N,) int64
        """
        self.eval()
        logits = self.forward(features)
        algo_ids = torch.argmax(logits, dim=1)
        return algo_ids


# ---------------------------------------------------------------------------
# 2. Feature Extraction (Conv2d-based, MobileNetV2-first but generic)
# ---------------------------------------------------------------------------

@dataclass
class LayerFeature:
    name: str
    features: List[float]


def _is_conv2d(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv2d)


def extract_layer_shapes(
    model: nn.Module,
    input_size: Tuple[int, int] = (32, 32),
    in_channels: int = 3,
    device: str = "cpu",
) -> Dict[str, Dict[str, torch.Size]]:
    """
    Run a dummy forward pass with hooks on selected layers to capture input/output shapes.

    We hook on:
        - Conv2d
        - Linear
        - BatchNorm2d
        - ReLU / ReLU6 / GELU
        - MaxPool2d / AvgPool2d

    Returns:
        shape_dict: {
          layer_name: {
            "in":  torch.Size([...]),
            "out": torch.Size([...]),
          }, ...
        }
    """
    model = model.to(device)
    model.eval()

    shape_dict: Dict[str, Dict[str, torch.Size]] = {}
    hooks = []

    # Register hooks for selected modules
    for name, module in model.named_modules():
        if isinstance(module, HOOK_LAYER_TYPES):

            def make_hook(layer_name: str):
                def hook(mod, inp, out):
                    # inp: tuple(tensor, ...), out: tensor or tuple
                    x_in = inp[0]
                    x_out = out
                    # out이 tuple일 수 있으니 첫 요소만 가져오는 방어 코드
                    if isinstance(x_out, (tuple, list)):
                        x_out = x_out[0]
                    shape_dict[layer_name] = {
                        "in": x_in.shape,
                        "out": x_out.shape,
                    }
                return hook

            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # Dummy forward
    dummy = torch.randn(1, in_channels, input_size[0], input_size[1], device=device)
    with torch.no_grad():
        _ = model(dummy)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    return shape_dict


def build_layer_features_generic(
    model: nn.Module,
    shapes: Dict[str, Dict[str, torch.Size]],
) -> List[LayerFeature]:

    features_list = []

    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        if name not in shapes:
            continue

        in_shape = shapes[name]["in"]
        out_shape = shapes[name]["out"]

        # --- Shape normalization ---
        # Conv2d: (N, C, H, W)
        # Linear: (N, Features)
        # Other: handle safely
        if len(in_shape) == 4:
            _, c_in, h_in, w_in = in_shape
        elif len(in_shape) == 2:
            # Linear-like layer
            _, c_in = in_shape
            h_in, w_in = 1.0, 1.0
        else:
            # Unsupported or weird shape → skip or treat as generic 1x1 op
            # You may choose to continue or skip
            print(f"[WARN] Unsupported in_shape for {name}: {in_shape}, skipping.")
            continue

        # Same for out_shape
        if len(out_shape) == 4:
            _, c_out, h_out, w_out = out_shape
        elif len(out_shape) == 2:
            _, c_out = out_shape
            h_out, w_out = 1.0, 1.0
        else:
            print(f"[WARN] Unsupported out_shape for {name}: {out_shape}, skipping.")
            continue


        # 1. Conv2d
        if isinstance(module, nn.Conv2d):
            k_h, k_w = module.kernel_size
            s_h, s_w = module.stride
            groups = module.groups
            has_bias = 1.0 if module.bias is not None else 0.0

            if groups == c_in and c_in == c_out:
                op_type = 1.0
            elif groups > 1:
                op_type = 2.0
            else:
                op_type = 0.0

            feats = [
                op_type, float(c_in), float(h_in), float(w_in),
                float(c_out), float(h_out), float(w_out),
                float(k_h), float(k_w),
                float(s_h), float(s_w),
                float(groups),
                has_bias,
            ]

        # 2. BatchNorm
        elif isinstance(module, nn.BatchNorm2d):
            feats = [
                10.0, float(c_in), float(h_in), float(w_in),
                float(c_out), float(h_out), float(w_out),
                0, 0, 0, 0,
                0, 0,
            ]

        # 3. Activation
        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU)):
            feats = [
                11.0, float(c_in), float(h_in), float(w_in),
                float(c_out), float(h_out), float(w_out),
                0, 0, 0, 0,
                0, 0,
            ]

        # 4. Pooling
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            k = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            s = module.stride if isinstance(module.stride, tuple) else (1, 1)

            feats = [
                12.0, float(c_in), float(h_in), float(w_in),
                float(c_out), float(h_out), float(w_out),
                float(k[0]), float(k[1]),
                float(s[0]), float(s[1]),
                0, 0,
            ]

        # 5. Linear
        elif isinstance(module, nn.Linear):
            feats = [
                20.0,
                float(c_in), 1.0, 1.0,
                float(c_out), 1.0, 1.0,
                0, 0, 0, 0,
                0,
                1.0 if module.bias is not None else 0.0,
            ]

        else:
            continue

        features_list.append(LayerFeature(name, feats))

    return features_list


def extract_layer_features_mobilenet_v2(
    model: nn.Module,
    input_size: Tuple[int, int] = (32, 32),
    in_channels: int = 3,
    device: str = "cpu",
) -> Tuple[List[List[float]], List[str]]:
    """
    MobileNetV2 용 기본 feature extractor지만 Conv2d 기반 모델이라면 대부분 재사용 가능.

    Steps:
        1. Conv2d 레이어에 hook 걸고 dummy forward로 in/out shape 수집
        2. shape + conv 파라미터로 feature 벡터 생성

    Returns:
        features:   List of [feature_dim] 리스트
        layer_names: 각 feature에 대응하는 layer path 이름 리스트
    """
    shapes = extract_layer_shapes(
        model=model,
        input_size=input_size,
        in_channels=in_channels,
        device=device,
    )
    lf_list = build_layer_features_generic(model, shapes)

    features = [lf.features for lf in lf_list]
    names = [lf.name for lf in lf_list]
    return features, names


# ---------------------------------------------------------------------------
# 3. Algo Selection Builder
# ---------------------------------------------------------------------------

def build_algo_selection(
    layer_names: List[str],
    feature_vectors: List[List[float]],
    selector: GenericLayerSelector,
    device: str = "cpu",
    include_features: bool = False,
) -> Dict:

    feats_tensor = torch.tensor(feature_vectors, dtype=torch.float32, device=device)
    
    algo_ids = selector.predict_algorithms(feats_tensor).cpu().tolist()

    # ✔ 알고리즘 번호 매핑
    layers_map = {
        name: int(aid)   # aid ∈ {0, 1, 2}
        for name, aid in zip(layer_names, algo_ids)
    }

    algo_selection = {
        "layers": layers_map,
        "meta": {
            "input_dim": selector.input_dim,
            "num_algos": selector.num_algos,
        },
    }

    if include_features:
        algo_selection["layer_features"] = {
            name: fv for name, fv in zip(layer_names, feature_vectors)
        }

    return algo_selection




def save_algo_selection_json(algo_selection: Dict, path: str) -> None:
    """
    Save algo_selection dict as JSON file.
    """
    import json
    with open(path, "w") as f:
        json.dump(algo_selection, f, indent=2)


# ---------------------------------------------------------------------------
# 4. Example / Utility: run with MobileNetV2 (optional)
# ---------------------------------------------------------------------------
'''
if __name__ == "__main__":
    """
    Optional quick test:
        - Try to import torchvision
        - Build MobileNetV2
        - Extract features
        - Run selector with random weights
        - Save algo_selection_mobilenet_v2.json

    NOTE: This is meant as a sanity-check script, not as a trained selector.
    """
    try:
        from torchvision.models import mobilenet_v2
    except ImportError:
        print("[Layer_Selector] torchvision not installed; skipping __main__ demo")
        exit(0)

    print("[Layer_Selector] Running demo for MobileNetV2...")

    device = "cpu"

    # 1) Build model
    model = mobilenet_v2(weights=None)  # or weights="IMAGENET1K_V1" if available

    # 2) Extract features
    features, names = extract_layer_features_mobilenet_v2(
        model, input_size=(32, 32), in_channels=3, device=device
    )
    print(f"[Layer_Selector] Extracted {len(features)} conv layers")

    if not features:
        print("[Layer_Selector] No features extracted; exiting.")
        exit(0)

    # 3) Build selector (untrained, random init)
    input_dim = len(features[0])
    selector = GenericLayerSelector(input_dim=input_dim, num_algos=3)
    selector.to(device)

    # 4) Build algo_selection
    algo_selection = build_algo_selection(
        layer_names=names,
        feature_vectors=features,
        selector=selector,
        device=device,
        include_features=True,
    )

    # 5) Save JSON
    out_path = "algo_selection_mobilenet_v2_demo.json"
    save_algo_selection_json(algo_selection, out_path)
    print(f"[Layer_Selector] Saved algo_selection to {out_path}")
'''