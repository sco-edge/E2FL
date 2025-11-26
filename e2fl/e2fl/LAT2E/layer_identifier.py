# predictor/layer_identifier.py

import math
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# 어떤 레이어를 "key"로 볼지 / "non-key"로 볼지 정의
KEY_LAYERS = (
    nn.Conv2d,
    nn.Linear,
    nn.MultiheadAttention,   # PyTorch standard
)

NON_KEY_LAYERS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.ReLU,
    nn.ReLU6,
    nn.GELU,
    nn.SiLU,
    nn.LeakyReLU,
    nn.LayerNorm,
    nn.GroupNorm,
)


def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return int(n)


def conv2d_flops_from_shapes(
    layer: nn.Conv2d,
    in_shape: Tuple[int, int, int, int],
    out_shape: Tuple[int, int, int, int],
) -> int:
    """
    Conv2d FLOPs (MACs * 2 안 쓰고, 그냥 MACs 수만 기준).
    in_shape: (N, C_in, H_in, W_in)
    out_shape: (N, C_out, H_out, W_out)
    """
    N, Cin, Hin, Win = in_shape
    N2, Cout, Hout, Wout = out_shape
    assert N == N2, "Batch size mismatch in conv FLOPs calc"

    Kh, Kw = layer.kernel_size
    groups = layer.groups

    if groups == 1:
        # standard conv
        flops = N * Cout * Hout * Wout * (Cin * Kh * Kw)
    else:
        # depthwise or grouped conv
        # depthwise: groups == Cin == Cout
        flops = N * Cout * Hout * Wout * (Cin * Kh * Kw // groups)

    return int(flops)


def linear_flops_from_shapes(
    layer: nn.Linear,
    in_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
) -> int:
    """
    Linear FLOPs 계산. in_shape, out_shape는 터진 batch dimension 기준.
    예: in_shape = (N, D_in), out_shape = (N, D_out)
    """
    if len(in_shape) == 1:
        # (D,) -> (D_out,) 같은 경우 batch=1 간주
        N = 1
        D_in = in_shape[0]
    else:
        N = in_shape[0]
        D_in = int(_numel(in_shape[1:]))

    D_out = layer.out_features
    flops = N * D_in * D_out
    return int(flops)


def mha_flops_from_shapes(
    layer: nn.MultiheadAttention,
    in_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
) -> int:
    """
    MultiheadAttention FLOPs rough model.
    PyTorch MHA: (L, N, E) or (N, L, E) depending on batch_first
    여기서는 attention score 계산 + projection 위주로 근사.
    """
    # shape 해석
    if layer.batch_first:
        # (N, L, E)
        N, L, E = in_shape[:3]
    else:
        # (L, N, E)
        L, N, E = in_shape[:3]

    h = layer.num_heads
    d_head = E // h

    # Q,K,V projections: 3 * N * L * E * E
    proj_qkv = 3 * N * L * E * E

    # Attention score: N * h * L * L * d_head
    attn_scores = N * h * L * L * d_head

    # Output projection: N * L * E * E
    out_proj = N * L * E * E

    return int(proj_qkv + attn_scores + out_proj)

def compute_C_key_C_non(layer_info):
    C_key = []
    C_non = []
    for layer in layer_info["layers"]:
        flops = layer.get("flops", 0)
        t = layer.get("type", "").lower()

        # key layers = conv / linear
        if "conv" in t or "linear" in t:
            C_key.append(flops)
        else:
            C_non.append(flops)

    return C_key, C_non


class LayerIdentifier:
    """
    Robust Layer Identifier:
    - forward hook으로 각 leaf module의 (input_shape, output_shape) 수집
    - Conv/Linear/Attention/BN/ReLU 등 FLOPs 계산
    - key/non-key 분리, C_key/C_non 계산
    - layers 메타데이터까지 함께 반환

    사용 예:
        model = mobilenet_v2(weights=None)
        identifier = LayerIdentifier(model, input_shape=(1, 3, 224, 224), model_name="mobilenet_v2")
        profile = identifier.analyze()
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, int, int, int],
        model_name: str = "unknown",
        is_transformer: bool = False,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.input_shape = input_shape
        self.is_transformer = is_transformer

        # hook에서 채우는 dict: name -> { "in_shape": ..., "out_shape": ... }
        self._io_shapes: Dict[str, Dict[str, Tuple[int, ...]]] = {}
        self._hooks: List[Any] = []

    # ------------------------------------------------------------------
    # Hook 등록/해제
    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        """각 leaf module에 대해 forward hook 등록."""

        def hook_fn(module: nn.Module, inputs, output):
            # module.__repr__과 이름 매핑을 위해 named_modules 사용
            # 그래서 여기서는 module의 id로 임시 저장
            # 실제 이름 매핑은 나중에 named_modules 순회 때 수행
            module_id = id(module)

            # 입력/출력 shape 잡기
            def _to_shape(x):
                if isinstance(x, torch.Tensor):
                    return tuple(x.shape)
                elif isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
                    # 첫 번째 텐서 기준
                    return tuple(x[0].shape)
                else:
                    return None

            in_shape = _to_shape(inputs[0]) if len(inputs) > 0 else None
            out_shape = _to_shape(output)

            if in_shape is None or out_shape is None:
                return

            if module_id not in self._io_shapes:
                self._io_shapes[module_id] = {}
            self._io_shapes[module_id]["in_shape"] = in_shape
            self._io_shapes[module_id]["out_shape"] = out_shape

        # leaf module에만 hook 등록
        for m in self.model.modules():
            if len(list(m.children())) == 0:
                self._hooks.append(m.register_forward_hook(hook_fn))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ------------------------------------------------------------------
    # FLOPs 계산 루틴
    # ------------------------------------------------------------------
    def _compute_flops_for_module(
        self,
        name: str,
        module: nn.Module,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
    ) -> Tuple[bool, int]:
        """
        해당 module이 key인지/아닌지, 그리고 FLOPs를 반환.
        (is_key, flops)
        """
        # Conv2d
        if isinstance(module, nn.Conv2d):
            flops = conv2d_flops_from_shapes(module, in_shape, out_shape)
            return True, flops

        # Linear
        if isinstance(module, nn.Linear):
            flops = linear_flops_from_shapes(module, in_shape, out_shape)
            return True, flops

        # MultiheadAttention
        if isinstance(module, nn.MultiheadAttention):
            flops = mha_flops_from_shapes(module, in_shape, out_shape)
            return True, flops

        # Non-key layers (BN, ReLU, etc.)
        if isinstance(module, NON_KEY_LAYERS):
            # 근사: 요소 개수만큼 연산
            flops = _numel(out_shape)
            return False, flops

        # 나머지 모듈들은 일단 non-key, FLOPs=0 처리 (필요시 여기 확장)
        return False, 0

    # ------------------------------------------------------------------
    # 분석 메인 함수
    # ------------------------------------------------------------------
    def analyze(self) -> Dict[str, Any]:
        """
        모델에 dummy input을 흘려보내 hook으로 shape 수집 후,
        layer별 FLOPs와 key/non-key 구분 결과를 반환.
        """
        self.model.eval()
        self._io_shapes.clear()

        # 1) hook 등록
        self._register_hooks()

        # 2) dummy forward (실제 input_shape대로)
        with torch.no_grad():
            dummy = torch.randn(*self.input_shape)
            _ = self.model(dummy)

        # 3) hook 제거
        self._remove_hooks()

        # 4) named_modules 순회하면서 layer별로 FLOPs 계산
        key_layers: List[Dict[str, Any]] = []
        non_key_layers: List[Dict[str, Any]] = []
        layers_summary: List[Dict[str, Any]] = []

        C_key_list: List[int] = []
        C_non_total = 0

        layer_idx = 0
        moduleid_to_name: Dict[int, str] = {}

        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue  # container skip

            mid = id(module)
            if mid not in self._io_shapes:
                # hook에서 shape을 못 얻은 레이어 (예: dropout, identity 등)
                continue

            in_shape = self._io_shapes[mid]["in_shape"]
            out_shape = self._io_shapes[mid]["out_shape"]

            is_key, flops = self._compute_flops_for_module(name, module, in_shape, out_shape)

            layer_info = {
                "idx": layer_idx,
                "name": name,
                "type": module.__class__.__name__,
                "is_key": bool(is_key),
                "flops": int(flops),
                "in_shape": list(in_shape),
                "out_shape": list(out_shape),
            }

            layers_summary.append(layer_info)
            
            if is_key and flops > 0:
                key_layers.append(
                    {
                        "idx": layer_idx,
                        "name": name,
                        "type": module.__class__.__name__,
                        "flops": int(flops),
                    }
                )
                C_key_list.append(int(flops))
            elif (not is_key) and flops > 0:
                non_key_layers.append(
                    {
                        "idx": layer_idx,
                        "name": name,
                        "type": module.__class__.__name__,
                        "flops": int(flops),
                    }
                )
                C_non_total += int(flops)

            layer_idx += 1
            moduleid_to_name[mid] = name

        # 최종 dict (model_profile_dict로 바로 사용 가능)
        profile: Dict[str, Any] = {
            "model_name": self.model_name,
            "input_shape": list(self.input_shape),

            "key_layers": key_layers,
            "non_key_layers": non_key_layers,

            "layers": layers_summary,

            # LATTE 표준
            "C_key": C_key_list,         # 기존 C_key_list
            "C_non": C_non_total,        # 기존 C_non_total

            # 유지 (있어도 상관 없음)
            "C_key_list": C_key_list,
            "C_key_total": int(sum(C_key_list)),
            "C_non_total": int(C_non_total),
        }


        return profile
