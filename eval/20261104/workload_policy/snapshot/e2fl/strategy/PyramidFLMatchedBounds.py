"""Experiment-only bounds adapter for the existing PyramidFL-style policy.

The speed estimate, ratio, and rounding rules stay in PyramidFL.py. This
adapter only enforces the shared physical sample bounds after that decision.
"""

from __future__ import annotations

from e2fl.strategy.PyramidFL import PyramidFLLoRA_NoDropout


class PyramidFLLoRA_MatchedBounds(PyramidFLLoRA_NoDropout):
    def __init__(self, *, pi_max_quota: int = 4,
                 jetson_max_quota: int = 60, **kwargs):
        if int(pi_max_quota) < 1 or int(jetson_max_quota) < 1:
            raise ValueError("matched sample caps must be positive")
        super().__init__(**kwargs)
        # Keep the hard caps separate from FedAvgLoRA's policy fields, which
        # are refreshed from each round's ConfigRecord.
        self.matched_pi_max_quota = int(pi_max_quota)
        self.matched_jetson_max_quota = int(jetson_max_quota)

    def _set_device_workload(self, cfg, device_name: str,
                             quota: int, epochs: int) -> None:
        if device_name == "RPi5":
            quota = min(int(quota), self.matched_pi_max_quota)
        elif device_name in ("jetson_orin_nx", "jetson_orin_nano_super"):
            quota = min(int(quota), self.matched_jetson_max_quota)
        super()._set_device_workload(cfg, device_name, quota, epochs)
