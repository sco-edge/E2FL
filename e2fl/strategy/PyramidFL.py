# e2fl/strategy/PyramidFL.py
"""
PyramidFL-LoRA-noDropout baseline strategy.

This is an adapted baseline for LoRA-based federated fine-tuning.
It keeps the same client-side protocol as FedAvgLoRA/LoRACast and only
changes server-side per-device workload allocation.

What it implements:
- PyramidFL-style adaptive local training:
  faster device groups receive larger local workload based on recent observed speed.

What it intentionally excludes:
- PyramidFL update dropout / parameter dropout.
  This baseline isolates workload allocation because LoRA updates are already compact
  and because update dropout would require a different partial-update upload protocol.

Compatibility:
- Works with the existing client code that already supports:
  max_samples_per_round_<device>
  local_epochs_<device>
- Reuses FedAvgLoRA logging, aggregation, energy, token, and network metrics.
"""

from __future__ import annotations

import math
from logging import INFO
from typing import Any

from flwr.app import ConfigRecord
from flwr.common import log

from e2fl.strategy.FedAvgLoRA import FedAvgLoRA


class PyramidFLLoRA_NoDropout(FedAvgLoRA):
    """PyramidFL-style adaptive local training baseline for federated LoRA.

    This strategy is meant to be compared against:
      - FedAvgLoRA: same workload for all clients
      - LoRACast: prediction-based round-window-aware workload allocation

    The policy is intentionally simple:
      speed_i = recent_work_i / recent_train_time_i
      scale_i = speed_i / min_speed
      workload_i = base_workload * clipped(scale_i)

    It does not use LoRACast's round-window prediction, energy estimator,
    slow-device rebalancing, or deadline guard.
    """

    def __init__(
        self,
        base_quota: int | None = None,
        base_epochs: int | None = None,
        max_scale: float = 4.0,
        speed_window: int = 5,
        use_tokens_for_speed: bool = True,
        scale_epochs_after_quota: bool = False,
        max_epochs: int = 4,
        min_quota: int | None = None,
        **kwargs: Any,
    ):
        # Disable LoRACast-specific controls inherited from FedAvgLoRA.
        kwargs["enable_rpi_only_rounds"] = False
        kwargs["use_token_ratio"] = False
        kwargs["enable_deadline_guard"] = False

        super().__init__(**kwargs)

        # If None, resolve from train_config at runtime so this baseline starts
        # from the same workload as FedAvg-LoRA/LoRACast.
        self.base_quota = int(base_quota) if base_quota is not None else None
        self.override_base_epochs = base_epochs
        self.max_scale = float(max_scale)
        self.speed_window = int(speed_window)
        self.use_tokens_for_speed = bool(use_tokens_for_speed)
        self.scale_epochs_after_quota = bool(scale_epochs_after_quota)
        self.max_epochs = int(max_epochs)
        self.min_quota = int(min_quota) if min_quota is not None else None

    def update_train_config(self, server_round: int, train_config: ConfigRecord) -> ConfigRecord:
        """Inject PyramidFL-LoRA-noDropout per-device workload config."""

        # First, reuse parent defaults and LoRA-rank propagation.
        # Since all LoRACast controls are disabled in __init__, this will not enable
        # rebalancing/window/deadline behaviors.
        train_config = super().update_train_config(server_round, train_config)

        train_config["round_mode"] = "PYRAMIDFL_LORA_NODROPOUT"

        # Use explicit base_epochs if provided, otherwise inherit global local_epochs.
        if self.override_base_epochs is None:
            try:
                base_epochs = int(train_config.get("local_epochs", 1) or 1)
            except Exception:
                base_epochs = 1
        else:
            base_epochs = int(self.override_base_epochs)

        # Resolve the common starting sample quota from the current config.
        # This keeps PyramidFL-LoRA-noDropout aligned with the exact workload
        # used by FedAvg-LoRA/LoRACast in the same TOML configuration.
        base_quota = self._resolve_base_quota(train_config)

        # Cold start: no speed history yet, so everyone gets the same workload.
        if server_round == 1 or not self.round_stats:
            self._set_all_devices_same(train_config, base_quota, base_epochs)
            self._log_round_config(server_round, train_config)
            return train_config

        speeds = {
            "RPi5": self._estimate_device_speed("RPi5"),
            "jetson_orin_nx": self._estimate_device_speed("jetson_orin_nx"),
            "jetson_orin_nano_super": self._estimate_device_speed("jetson_orin_nano_super"),
        }

        valid_speeds = [v for v in speeds.values() if v > 0.0]
        if not valid_speeds:
            self._set_all_devices_same(train_config, base_quota, base_epochs)
            self._log_round_config(server_round, train_config)
            return train_config

        ref_speed = min(valid_speeds)

        for dev_name in ("RPi5", "jetson_orin_nx", "jetson_orin_nano_super"):
            speed = speeds.get(dev_name, 0.0)
            if speed <= 0.0:
                scale = 1.0
            else:
                scale = speed / max(ref_speed, 1e-12)

            scale = max(1.0, min(self.max_scale, scale))
            quota, epochs = self._scale_to_workload(base_quota, base_epochs, scale)
            self._set_device_workload(train_config, dev_name, quota, epochs)

        self._log_round_config(server_round, train_config, speeds=speeds)
        return train_config

    def _resolve_base_quota(self, cfg: ConfigRecord) -> int:
        """Resolve the starting sample quota from config, falling back safely.

        Priority:
          1) explicit constructor base_quota
          2) device-specific RPi5 quota already populated by the parent strategy
          3) global max_samples_per_round from TOML
          4) inherited rpi_min_quota
          5) conservative default 256
        """
        if self.base_quota is not None:
            return int(self.base_quota)

        for key in (
            "max_samples_per_round_RPi5",
            "max_samples_per_round",
        ):
            value = self._parse_positive_int(cfg.get(key, None))
            if value is not None:
                return value

        value = self._parse_positive_int(getattr(self, "rpi_min_quota", None))
        if value is not None:
            return value

        return 256

    def _parse_positive_int(self, value) -> int | None:
        try:
            if value in (None, ""):
                return None
            parsed = int(value)
            if parsed > 0:
                return parsed
        except Exception:
            return None
        return None

    def _estimate_device_speed(self, device_name: str) -> float:
        """Estimate recent device-group speed from committed round_stats.

        Returns work/sec where work is either input tokens or examples.
        """
        vals: list[float] = []

        for rnd in sorted(self.round_stats.keys(), reverse=True)[: self.speed_window]:
            rs = self.round_stats.get(rnd, {})
            entry = rs.get(device_name, {}) if isinstance(rs, dict) else {}
            if not entry:
                continue

            try:
                t = float(entry.get("train_time", 0.0) or 0.0)
            except Exception:
                t = 0.0

            if self.use_tokens_for_speed:
                try:
                    work = float(entry.get("tokens", 0.0) or 0.0)
                except Exception:
                    work = 0.0
            else:
                try:
                    work = float(entry.get("num_examples", 0.0) or 0.0)
                except Exception:
                    work = 0.0

            if t > 0.0 and work > 0.0:
                vals.append(work / t)

        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def _scale_to_workload(self, base_quota: int, base_epochs: int, scale: float) -> tuple[int, int]:
        """Convert continuous speed scale into sample quota and epochs.

        Default mode:
          - scale only sample quota, keep epochs fixed.
          - This is the cleanest workload-allocation baseline.

        Optional mode:
          - grow quota up to max_scale*base_quota, then grow epochs.
          - Disabled by default to keep comparison simple.
        """
        scale = max(1.0, min(float(scale), self.max_scale))
        min_quota = self.min_quota if self.min_quota is not None else int(base_quota)

        if not self.scale_epochs_after_quota:
            quota = int(round(base_quota * scale))
            quota = max(min_quota, quota)
            return quota, int(base_epochs)

        # Optional two-level scaling: quota first, then epochs.
        quota_scale = min(scale, self.max_scale)
        quota = int(round(base_quota * quota_scale))
        quota = max(min_quota, quota)

        # Conservative epoch scaling; usually keep this off for clean comparison.
        epoch_scale = max(1.0, scale / max(self.max_scale, 1.0))
        epochs = int(math.ceil(base_epochs * epoch_scale))
        epochs = max(1, min(self.max_epochs, epochs))
        return quota, epochs

    def _set_all_devices_same(self, cfg: ConfigRecord, quota: int, epochs: int) -> None:
        for dev_name in ("RPi5", "jetson_orin_nx", "jetson_orin_nano_super"):
            self._set_device_workload(cfg, dev_name, quota, epochs)

    def _set_device_workload(self, cfg: ConfigRecord, device_name: str, quota: int, epochs: int) -> None:
        quota = int(max(0, quota))
        epochs = int(max(0, epochs))

        if device_name == "RPi5":
            cfg["max_samples_per_round_RPi5"] = quota
            cfg["local_epochs_RPi5"] = epochs
        elif device_name == "jetson_orin_nx":
            cfg["max_samples_per_round_jetson_orin_nx"] = quota
            cfg["local_epochs_jetson_orin_nx"] = epochs
        elif device_name == "jetson_orin_nano_super":
            cfg["max_samples_per_round_jetson_orin_nano_super"] = quota
            cfg["local_epochs_jetson_orin_nano_super"] = epochs
        else:
            # Fallback for unknown device names: do not break training.
            cfg["max_samples_per_round"] = quota
            cfg["local_epochs"] = epochs

    def _log_round_config(
        self,
        server_round: int,
        cfg: ConfigRecord,
        speeds: dict[str, float] | None = None,
    ) -> None:
        speeds = speeds or {}
        log(
            INFO,
            "[PyramidFLLoRA_NoDropout][ROUND %s] "
            "RPi5=(S=%s,E=%s,speed=%.4f), "
            "NX=(S=%s,E=%s,speed=%.4f), "
            "Nano=(S=%s,E=%s,speed=%.4f)",
            server_round,
            cfg.get("max_samples_per_round_RPi5", ""),
            cfg.get("local_epochs_RPi5", ""),
            float(speeds.get("RPi5", 0.0) or 0.0),
            cfg.get("max_samples_per_round_jetson_orin_nx", ""),
            cfg.get("local_epochs_jetson_orin_nx", ""),
            float(speeds.get("jetson_orin_nx", 0.0) or 0.0),
            cfg.get("max_samples_per_round_jetson_orin_nano_super", ""),
            cfg.get("local_epochs_jetson_orin_nano_super", ""),
            float(speeds.get("jetson_orin_nano_super", 0.0) or 0.0),
        )
