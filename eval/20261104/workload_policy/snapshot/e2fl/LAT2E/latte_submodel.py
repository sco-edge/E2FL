"""
latte_submodel.py — LATTE submodel allocator

This module implements:
  • SubModelConfig: a dataclass representing a single submodel (windowed/width-scaled).
  • Width-scaling submodel generation (generate_width_submodels).
  • Submodel selection (select_best_submodel), choosing the largest feasible submodel
    under a latency deadline using LatteFinePredictor and a base workload profile.

Usage:
  • Server loads a pre-generated submodel table (offline).
  • Client loads selector + predictor + submodel table.
  • Client computes current bandwidth/memory, predicts T_total for each submodel,
    and selects the best one via select_best_submodel.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Iterable, Optional
import json
import math
import torch.nn as nn

from e2fl.LAT2E.latte_predictor import LatteFinePredictor


ChannelWindow = Tuple[int, int]
LayerWindows = Dict[str, List[ChannelWindow]]


@dataclass
class SubModelConfig:
    id: int
    windows: LayerWindows
    scale_factor: float
    T_single: Optional[float] = None
    mem_est: Optional[float] = None
    model_size_bytes: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SubModelConfig":
        return cls(
            id=int(d["id"]),
            windows={k: [tuple(w) for w in v] for k, v in d.get("windows", {}).items()},
            scale_factor=float(d.get("scale_factor", 1.0)),
            T_single=(float(d["T_single"]) if d.get("T_single") is not None else None),
            mem_est=(float(d["mem_est"]) if d.get("mem_est") is not None else None),
            model_size_bytes=(
                int(d["model_size_bytes"]) if d.get("model_size_bytes") is not None else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["windows"] = {k: [list(w) for w in v] for k, v in self.windows.items()}
        return d


# ----------------------------------------------------------------------------
# Helper: extract conv output channels from a PyTorch model
# ----------------------------------------------------------------------------

def get_conv_output_channels(model: nn.Module) -> Dict[str, int]:
    conv_channels: Dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and len(list(module.children())) == 0:
            conv_channels[name] = int(module.out_channels)
    return conv_channels


# ----------------------------------------------------------------------------
# Submodel window building
# ----------------------------------------------------------------------------

def _build_windows_for_scale(conv_channels: Dict[str, int], scale: float) -> LayerWindows:
    windows: LayerWindows = {}
    for layer_name, c_out in conv_channels.items():
        used = max(1, int(math.floor(c_out * scale)))
        used = min(used, c_out)
        windows[layer_name] = [(0, used)]
    return windows


def generate_width_submodels(
    conv_channels: Dict[str, int],
    scales: Optional[List[float]] = None,
    base_model_size_bytes: Optional[int] = None,
) -> List[SubModelConfig]:

    if scales is None:
        scales = [1.0, 0.75, 0.5, 0.35, 0.25]

    submodels: List[SubModelConfig] = []
    next_id = 0

    for scale in scales:
        windows = _build_windows_for_scale(conv_channels, scale)

        if base_model_size_bytes is not None:
            approx_scale = scale * scale
            model_size_bytes = int(base_model_size_bytes * approx_scale)
        else:
            model_size_bytes = None

        sub_cfg = SubModelConfig(
            id=next_id,
            windows=windows,
            scale_factor=float(scale),
            T_single=None,
            mem_est=None,
            model_size_bytes=model_size_bytes,
        )
        submodels.append(sub_cfg)
        next_id += 1
    return submodels


# ----------------------------------------------------------------------------
# JSON I/O
# ----------------------------------------------------------------------------

def save_submodel_table(model_name: str, submodels: Iterable[SubModelConfig], path: str) -> None:
    data = {
        "model": model_name,
        "submodels": [s.to_dict() for s in submodels],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_submodel_table(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        raw = json.load(f)
    subs = [SubModelConfig.from_dict(d) for d in raw.get("submodels", [])]
    return {
        "model": raw.get("model", ""),
        "submodels": subs,
    }


# ----------------------------------------------------------------------------
# Submodel selection
# ----------------------------------------------------------------------------


def select_best_submodel(
    submodels: List[SubModelConfig],
    predictor: LatteFinePredictor,
    algo_selection: List[str],
    C_key_full: List[float],
    C_non_full: float,
    B_bytes_per_sec: float,
    num_epochs: int,
    batch_size: int,
    num_batches: int,
    deadline_U: float,
    base_param_size_bytes: int,
) -> Tuple[SubModelConfig, Dict[str, float]]:
    """
    Choose the largest feasible submodel whose predicted T_total <= deadline_U.
    Return (chosen_submodel, prediction_dict).

    Parameters
    ----------
    submodels:
        Candidate submodels (width-scaled configs).
    predictor:
        LatteFinePredictor (이미 디바이스별 betas를 내장).
    algo_selection, C_key_full, C_non_full:
        Full-model workload profile (e.g., mobilenet_v2_workload_fine.json).
    B_bytes_per_sec:
        Effective uplink bandwidth in bytes/second.
    num_epochs, batch_size, num_batches:
        Local training configuration.
    deadline_U:
        Round deadline.
    base_param_size_bytes:
        Full model parameter size in bytes.
    """

    if len(algo_selection) != len(C_key_full):
        raise ValueError("algo_selection and C_key_full must have the same length")

    # 큰 submodel부터 시도
    candidates = sorted(submodels, key=lambda x: -x.scale_factor)

    best_cfg: Optional[SubModelConfig] = None
    best_pred: Optional[Dict[str, float]] = None

    for s in candidates:
        scale2 = s.scale_factor * s.scale_factor

        # width scaling: FLOPs, non-key, 파라미터 모두 scale^2로 근사
        C_key_scaled = [c * scale2 for c in C_key_full]
        C_non_scaled = C_non_full * scale2
        param_size_bytes = int(base_param_size_bytes * scale2)

        sub_dict = {
            "algo_selection": algo_selection,
            "C_key": C_key_scaled,
            "C_non": C_non_scaled,
            "param_size_bytes": param_size_bytes,
            # 필요하면 여기 width_factors 같은 mask도 넣을 수 있음
        }

        pred = predictor.predict(
            submodel=sub_dict,
            B_bytes_per_sec=B_bytes_per_sec,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_batches=num_batches,
        )

        if pred.get("T_total", float("inf")) <= deadline_U:
            best_cfg = s
            best_pred = pred
            break

    # 어떤 submodel도 deadline을 만족 못 하면 가장 작은 걸 채택
    if best_cfg is None:
        best_cfg = sorted(submodels, key=lambda x: x.scale_factor)[0]

        scale2 = best_cfg.scale_factor * best_cfg.scale_factor
        C_key_scaled = [c * scale2 for c in C_key_full]
        C_non_scaled = C_non_full * scale2
        param_size_bytes = int(base_param_size_bytes * scale2)

        sub_dict = {
            "algo_selection": algo_selection,
            "C_key": C_key_scaled,
            "C_non": C_non_scaled,
            "param_size_bytes": param_size_bytes,
        }

        best_pred = predictor.predict(
            submodel=sub_dict,
            B_bytes_per_sec=B_bytes_per_sec,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_batches=num_batches,
        )

    if best_cfg is not None and best_pred is not None:
        try:
            best_cfg.T_single = float(best_pred.get("T_single", 0.0))
        except Exception:
            pass

    return best_cfg, best_pred

