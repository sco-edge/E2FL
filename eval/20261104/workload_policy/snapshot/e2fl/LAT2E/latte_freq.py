from __future__ import annotations

"""
latte_freq.py

Frequency table (F) utilities for LATTE-style FL.

The frequency table tracks how often each channel of each key layer has been
trained. This metadata is used when selecting submodels so that rarely trained
parameters can be prioritized.
"""

from typing import Dict, List, Iterable, Tuple, Any

import copy
import math
import torch.nn as nn

from .latte_submodel import LayerWindows, get_conv_output_channels, SubModelConfig

# Type alias: layer_name -> [count per output channel]
FreqTable = Dict[str, List[int]]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def init_frequency_table_from_model(model: nn.Module) -> FreqTable:
    """Initialize F table with zeros based on Conv2d out_channels in `model`."""
    conv_channels = get_conv_output_channels(model)
    return init_frequency_table_from_conv_channels(conv_channels)


def init_frequency_table_from_conv_channels(conv_channels: Dict[str, int]) -> FreqTable:
    """Initialize F table from a mapping layer_name -> out_channels."""
    F: FreqTable = {}
    for layer_name, c_out in conv_channels.items():
        F[layer_name] = [0 for _ in range(int(c_out))]
    return F


# ---------------------------------------------------------------------------
# Update / merge
# ---------------------------------------------------------------------------

def update_frequency_table(F: FreqTable, windows: LayerWindows) -> None:
    """In-place update of F given the submodel's channel windows.

    Args:
        F: Frequency table to be updated.
        windows: Mapping layer_name -> list of (start, length) windows,
                 usually taken from SubModelConfig.windows.

    For each window, we increment F[layer][idx] by 1 for all channel indices
    that are part of the window (bounded by len(F[layer])).
    """
    for layer_name, win_list in windows.items():
        if layer_name not in F:
            continue
        counts = F[layer_name]
        n = len(counts)
        for start, length in win_list:
            start = int(start)
            length = int(length)
            end = min(start + length, n)
            for idx in range(max(0, start), end):
                counts[idx] += 1


def merge_frequency_tables(F_server: FreqTable, F_clients: Iterable[FreqTable]) -> FreqTable:
    """Merge multiple client F tables into the server's F table.

    The simple policy here is "sum across all clients", which approximates
    global training frequency. If you prefer averaging, you can divide by
    the number of clients afterwards.

    Returns:
        New F table (does not mutate the original F_server).
    """
    merged: FreqTable = copy.deepcopy(F_server)

    for F_c in F_clients:
        for layer_name, counts_c in F_c.items():
            if layer_name not in merged:
                merged[layer_name] = [0 for _ in counts_c]

            counts_s = merged[layer_name]
            # align length (extend with zeros if necessary)
            if len(counts_s) < len(counts_c):
                counts_s.extend([0] * (len(counts_c) - len(counts_s)))

            for i, val in enumerate(counts_c):
                counts_s[i] += int(val)

    return merged


# ---------------------------------------------------------------------------
# (De)serialization helpers
# ---------------------------------------------------------------------------

def serialize_frequency_table(F: FreqTable) -> Dict[str, Any]:
    """Prepare F for transport (e.g., via Flower metrics or Records).

    This returns a plain dict of {layer_name: [int, ...]}, which is JSON-safe.
    The caller is responsible for any compression or encoding (e.g., packing
    into bytes, numpy arrays, or an ArrayRecord) required by the FL framework.
    """
    return {layer: list(map(int, counts)) for layer, counts in F.items()}


def deserialize_frequency_table(obj: Dict[str, Any]) -> FreqTable:
    """Reconstruct an F table from a serialized dict.

    The input is expected to be {layer_name: [int, ...]}.
    """
    F: FreqTable = {}
    for layer, counts in obj.items():
        F[layer] = [int(v) for v in counts]
    return F

def _indices_to_windows(indices: List[int]) -> List[Tuple[int, int]]:
    """정렬된 채널 index 리스트를 (start, length) 윈도우 리스트로 압축.

    예: [0,1,2,4,5] -> [(0,3), (4,2)]
    """
    if not indices:
        return []

    indices = sorted(indices)
    windows: List[Tuple[int, int]] = []
    start = indices[0]
    prev = indices[0]

    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        # 구간 끊기
        windows.append((start, prev - start + 1))
        start = idx
        prev = idx

    windows.append((start, prev - start + 1))
    return windows


def assign_windows_for_submodel(
    F: FreqTable,
    conv_channels: Dict[str, int],
    sub_cfg: SubModelConfig,
) -> None:
    """주어진 sub_cfg.scale_factor에 맞춰,
    각 레이어에서 F가 낮은 채널부터 window를 배치.

    LATTE 논문에서 말하는:
      - 라운드마다 sub-model 크기(scale_factor)는 바뀔 수 있지만,
      - F가 작은(덜 훈련된) 채널들을 우선적으로 window에 포함시켜
        장기적으로는 파라미터를 고르게 학습시키는 아이디어를 구현.

    결과:
      - sub_cfg.windows 가 in-place로 갱신된다.
    """

    windows: LayerWindows = {}
    scale = float(sub_cfg.scale_factor)

    for layer_name, c_out in conv_channels.items():
        c_out = int(c_out)
        if c_out <= 0:
            continue

        # 이번 submodel에서 이 레이어에 사용할 채널 수
        k = max(1, int(math.floor(c_out * scale)))
        k = min(k, c_out)

        # 이 레이어에 대한 frequency 벡터
        counts = F.get(layer_name, [0] * c_out)
        if len(counts) < c_out:
            counts = counts + [0] * (c_out - len(counts))

        # F가 작은 채널 index부터 k개 선택
        idx_sorted = sorted(range(c_out), key=lambda i: counts[i])
        selected = sorted(idx_sorted[:k])

        # 연속된 index들을 (start, length) 윈도우로 묶기
        layer_windows = _indices_to_windows(selected)
        if layer_windows:
            windows[layer_name] = layer_windows

    sub_cfg.windows = windows