"""Explicit aggregation weights for LoRACast learning ablations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


MODES = {"processed_samples", "dataset_size", "uniform"}


def normalized_weights(
    metrics: Sequence[Mapping[str, object]],
    mode: str,
    *,
    processed_key: str = "num-examples",
) -> tuple[list[float], list[float]]:
    """Return raw and normalized weights over replies actually aggregated.

    ``dataset_size`` is the number of unique training records owned by a client,
    whereas ``num-examples`` counts examples executed in the current round and
    may count the same record again in another local epoch.
    """
    if mode not in MODES:
        raise ValueError(f"unsupported aggregation mode: {mode}")
    if not metrics:
        raise ValueError("no client metrics available for aggregation")
    key = {"processed_samples": processed_key, "dataset_size": "dataset_size"}.get(mode)
    raw = []
    for item in metrics:
        if mode == "uniform":
            value = 1.0
        else:
            if key not in item:
                raise ValueError(f"missing aggregation metric: {key}")
            try:
                value = float(item[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid aggregation metric: {key}") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"aggregation metric must be positive and finite: {key}")
        raw.append(value)
    total = sum(raw)
    return raw, [value / total for value in raw]
