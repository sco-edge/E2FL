#!/usr/bin/env python3
"""The official PyramidFL positive-slack iteration rule, isolated from its other features."""
from __future__ import annotations

import math


def choose_steps(profile: dict, eligible_sizes: dict[str, int]) -> dict:
    h0 = int(profile["base_optimizer_steps"])
    beta = float(profile["beta"])
    common_max = int(profile["maximum_optimizer_steps"])
    if h0 <= 0 or common_max < h0 or not 0 <= beta <= 1:
        raise ValueError("invalid fixed baseline constraints")
    devices = profile["devices"]
    if set(devices) != set(eligible_sizes) or not devices:
        raise ValueError("device profile and eligible data do not match")
    times = {}
    for name, values in devices.items():
        compute = float(values["base_compute_s"])
        noncompute = float(values["base_noncompute_completion_s"])
        if not (math.isfinite(compute) and compute > 0 and math.isfinite(noncompute) and noncompute >= 0):
            raise ValueError("invalid device completion profile")
        times[name] = compute + noncompute
    target = max(times.values())
    results = {}
    for name, values in devices.items():
        compute = float(values["base_compute_s"])
        slack = target - times[name]
        extra_iterations = math.floor(slack / (compute / h0))
        ratio = 1.0 + min(10.0, beta * extra_iterations / h0)
        original_requested = int(h0 * ratio)
        # Common LoRA device/data constraints.  DFull is large, so the
        # original learner's five-pass bound is inactive in this experiment.
        allowed = min(original_requested, common_max, 5 * int(eligible_sizes[name]))
        if allowed < h0 or allowed > int(eligible_sizes[name]):
            raise ValueError("E=1 fresh-data execution cannot realize requested iterations")
        results[name] = {
            "base_compute_s": compute,
            "base_noncompute_completion_s": float(values["base_noncompute_completion_s"]),
            "base_completion_s": times[name],
            "reference_completion_s": target,
            "slack_s": slack,
            "extra_iterations_before_beta": extra_iterations,
            "original_requested_steps": original_requested,
            "executed_target_steps": allowed,
            "S": allowed,
            "H": allowed,
            "E": 1,
        }
    return results
