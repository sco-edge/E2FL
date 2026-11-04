"""Isolated Fixed/A/B/C Nano-reference variants for the seed42 pilot.

The historical cost estimator and NX candidate search are inherited. All
policies receive the same measured S360/H0 calibration during rounds 1–4.
"""
from __future__ import annotations

import json
from pathlib import Path

from e2fl.strategy.DynamicSizingPilot import FrozenJetsonCost, LastCompletedStepRate
from e2fl.strategy.FedAvgLoRA import FedAvgLoRA
from e2fl.strategy.PyramidFL import PyramidFLLoRA_NoDropout
from nano_reference_strategy import NANO, NX, NanoReferenceLoRACast
from policy_resume import run_remaining_rounds


class CalibratedNanoPolicy(NanoReferenceLoRACast):
    policy_code = "unknown"
    fixed_after_calibration = False

    def __init__(self, **kwargs):
        super().__init__(participation_action="none", **kwargs)
        self.resume_completed_round = 0

    def start(self, *args, **kwargs):
        if not self.resume_completed_round:
            return super().start(*args, **kwargs)
        if args or "train_config" not in kwargs:
            raise ValueError("policy recovery requires the original keyword arguments")
        for key in ("reference-nano-samples", "reference-nano-steps",
                    "reference-nano-correction-max-samples",
                    "reference-nano-correction-max-steps",
                    "reference-nano-correction-max-seconds"):
            kwargs["train_config"][key] = self.nano_config[key]
        return run_remaining_rounds(
            self, completed_round=self.resume_completed_round, **kwargs
        )

    def update_train_config(self, server_round, train_config):
        config = super().update_train_config(server_round, train_config)
        round_number = int(server_round)
        native = {"samples": int(config[f"max_samples_per_round_{NX}"]),
                  "steps": int(config[f"local_steps_{NX}"])}
        calibrated = round_number <= 4
        frozen = self.fixed_after_calibration and round_number > 4
        if calibrated or frozen:
            device_map = self._known_client_device_map()
            nx_uid = next((uid for uid, name in device_map.items() if name == NX), None)
            self._set_workload(config, NX, nx_uid, 360, 0)
        requested_nano = (int(config[f"max_samples_per_round_{NANO}"]),
                          int(config[f"local_steps_{NANO}"]))
        requested_nx = (int(config[f"max_samples_per_round_{NX}"]),
                        int(config[f"local_steps_{NX}"]))
        if requested_nano != (320, 320) or not (360 <= requested_nx[0] <= 600):
            raise RuntimeError("policy request violates frozen common workload bounds")
        if frozen and requested_nx != (360, 0):
            raise RuntimeError("Fixed NX workload changed after calibration")
        record = {"round": round_number, "policy": self.policy_code,
                  "calibration_override": calibrated,
                  "fixed_override": frozen,
                  "native_nx_candidate": native,
                  "delivered_nano_samples_steps": requested_nano,
                  "delivered_nx_samples_steps": requested_nx,
                  "decision_uses_completed_rounds_only": True}
        target = Path(self.log_dir) / f"policy_effective_round_{round_number:02d}.json"
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return config


class NanoPolicyFixed(CalibratedNanoPolicy):
    policy_code = "F_calibrated_fixed"
    fixed_after_calibration = True


class NanoPolicyFrozen(FrozenJetsonCost, CalibratedNanoPolicy):
    policy_code = "A_frozen_NX_cost_after_round4"


class NanoPolicyLastRate(LastCompletedStepRate, CalibratedNanoPolicy):
    policy_code = "B_last_completed_step_rate"


class NanoPolicyOnline(CalibratedNanoPolicy):
    policy_code = "C_original_online_cost"


class AuditedBaseline:
    """Record delivered quotas and enable the same round-boundary state audit.

    This harness does not alter either baseline's speed estimator or search.
    """

    policy_code = "unknown"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resume_completed_round = 0

    def start(self, *args, **kwargs):
        if not self.resume_completed_round:
            return super().start(*args, **kwargs)
        if args or "train_config" not in kwargs:
            raise ValueError("baseline recovery requires the original keyword arguments")
        return run_remaining_rounds(
            self, completed_round=self.resume_completed_round, **kwargs
        )

    def update_train_config(self, server_round, train_config):
        config = super().update_train_config(server_round, train_config)
        requested = {
            dev: (int(config[f"max_samples_per_round_{dev}"]),
                  int(config[f"local_steps_{dev}"]))
            for dev in (NANO, NX)
        }
        self._validate_request(requested)
        record = {
            "round": int(server_round), "policy": self.policy_code,
            "delivered_nano_samples_steps": requested[NANO],
            "delivered_nx_samples_steps": requested[NX],
            "decision_uses_completed_rounds_only": True,
        }
        target = Path(self.log_dir) / f"policy_effective_round_{int(server_round):02d}.json"
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return config


class NanoUniform(AuditedBaseline, FedAvgLoRA):
    """Existing FedAvgLoRA with the same safe S=320 quota on both devices."""

    policy_code = "U_FedAvg_uniform"

    def update_train_config(self, server_round, train_config):
        # FedAvgLoRA's baseline branch reads this as the requested equal quota.
        train_config["max_samples_per_round"] = 320
        return super().update_train_config(server_round, train_config)

    @staticmethod
    def _validate_request(requested):
        if requested != {NANO: (320, 0), NX: (320, 0)}:
            raise RuntimeError(f"uniform baseline workload changed: {requested}")


class NanoPyramidBounded(AuditedBaseline, PyramidFLLoRA_NoDropout):
    """Existing PyramidFL-style rule under the common Nano/NX safety bounds."""

    policy_code = "P_PyramidFL_style_no_dropout"

    def _set_device_workload(self, cfg, device_name, quota, epochs):
        if device_name == NANO:
            quota = 320
        elif device_name == NX:
            quota = min(int(quota), 600)
        # The experimental comparison keeps a single local epoch everywhere.
        super()._set_device_workload(cfg, device_name, quota, 1)

    @staticmethod
    def _validate_request(requested):
        nano, nx = requested[NANO], requested[NX]
        if nano != (320, 0) or not (320 <= nx[0] <= 600 and nx[1] == 0):
            raise RuntimeError(f"PyramidFL workload outside common bounds: {requested}")
