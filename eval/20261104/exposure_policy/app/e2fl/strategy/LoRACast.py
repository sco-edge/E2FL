# e2fl/strategy/LoRACast.py
"""
Communication-aware LoRACast strategy.

This module adds LoRACast's communication-aware round-time prediction on top
of the project-owned workload-control base strategy:

    T_i = T_comp_i + T_comm_i
    T_comm_i = estimated_upload_bytes_i / recent_upload_rate_i

Design choices:
- Compute prediction is inherited from FedAvgLoRA.
- Communication prediction uses recently observed upload time and actual
  ArrayRecord tensor bytes received by the server.
- The existing compute-time calibration in FedAvgLoRA is preserved by disabling
  the communication term while _write_metrics() internally calibrates compute.
- No client changes are required.
"""

from __future__ import annotations

import math
from logging import INFO, WARNING
from typing import Any

import numpy as np

from flwr.app import ConfigRecord
from flwr.common import log

from e2fl.strategy.FedAvgLoRA import FedAvgLoRA


class LoRACastCommAware(FedAvgLoRA):
    """LoRACast with a lightweight upload-communication predictor.

    Use this strategy when network conditions should participate in workload allocation.

    The inherited LoRACast policy still performs the same candidate search, but
    now _predict_train_time() returns compute + communication time during policy
    decisions.
    """

    def __init__(
        self,
        comm_ema_alpha: float = 0.3,
        min_comm_time_s: float = 1e-3,
        min_upload_rate_Bps: float = 1.0,
        use_comm_prediction: bool = True,
        comm_safety_factor: float = 1.0,
        static_allocation_after_round: int = 0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.comm_ema_alpha = float(comm_ema_alpha)
        self.min_comm_time_s = float(min_comm_time_s)
        self.min_upload_rate_Bps = float(min_upload_rate_Bps)
        self.use_comm_prediction = bool(use_comm_prediction)
        self.comm_safety_factor = float(comm_safety_factor)
        self.static_allocation_after_round = int(static_allocation_after_round)
        self._static_workload: dict[str, int] | None = None

        # EMA upload throughput and payload size.
        # Keys can be device group names ("RPi5") or per-client keys ("RPi5:1").
        self.upload_rate_Bps: dict[str, float] = {}
        self.upload_bytes: dict[str, float] = {}

        # Internal guard: parent _write_metrics() calls _predict_train_time()
        # for compute-time calibration. During that period we must not add comm.
        self._disable_comm_in_prediction = False

    def update_train_config(self, server_round: int, train_config: ConfigRecord) -> ConfigRecord:
        train_config = super().update_train_config(server_round, train_config)
        cutoff = self.static_allocation_after_round
        if cutoff <= 0 or server_round <= cutoff:
            return train_config

        prefixes = ("max_samples_per_round_", "local_steps_", "local_epochs_")
        if self._static_workload is None:
            if train_config.get("round_mode") != "BOTH":
                raise RuntimeError("static allocation calibration requires a BOTH round")
            self._static_workload = {
                key: int(value) for key, value in train_config.items()
                if key.startswith(prefixes) and value not in ("", None)
            }
            log(INFO, "[LoRACastCommAware] static allocation captured at round %s: %s",
                server_round, self._static_workload)
        else:
            for key in list(train_config.keys()):
                if key.startswith(prefixes):
                    del train_config[key]
            train_config.update(self._static_workload)
            train_config["round_mode"] = "BOTH"
            log(INFO, "[LoRACastCommAware] static allocation reused in round %s: %s",
                server_round, self._static_workload)
        return train_config

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _predict_train_time(self, identity_key: str, num_examples: float, cfg: ConfigRecord) -> float:
        """Predict round-relevant training time.

        In policy mode:
          inherited compute prediction + estimated upload communication time

        During parent logging/calibration:
          inherited compute prediction only
        """
        comp_t = super()._predict_train_time(identity_key, num_examples, cfg)

        if (
            not self.use_comm_prediction
            or self._disable_comm_in_prediction
        ):
            return comp_t

        comm_t = self._predict_comm_time(identity_key)
        return comp_t + comm_t

    def _predict_comm_time(self, identity_key: str) -> float:
        """Predict upload communication time for the model update."""
        if not self.use_comm_prediction:
            return 0.0

        device_name = self._identity_to_device_name(identity_key)

        # Prefer per-client estimates. Fall back to device-group estimates.
        bytes_est = self.upload_bytes.get(identity_key)
        if bytes_est is None and device_name != identity_key:
            bytes_est = self.upload_bytes.get(device_name)
        if bytes_est is None:
            bytes_est = self._fallback_payload_bytes(device_name)

        rate = self.upload_rate_Bps.get(identity_key)
        if rate is None and device_name != identity_key:
            rate = self.upload_rate_Bps.get(device_name)

        if bytes_est is None or bytes_est <= 0.0:
            return 0.0
        if rate is None or rate <= 0.0:
            return 0.0

        rate = max(float(rate), self.min_upload_rate_Bps)
        return self.comm_safety_factor * float(bytes_est) / rate

    def _identity_to_device_name(self, identity_key: str) -> str:
        if isinstance(identity_key, str) and ":" in identity_key:
            return identity_key.split(":", 1)[0]
        return str(identity_key)

    def _fallback_payload_bytes(self, device_name: str) -> float | None:
        """Best-effort fallback before any upload observation exists.

        If model_info includes LoRA parameter counts, use it. Otherwise return None
        and let the first round run compute-only.
        """
        try:
            mi = getattr(self, "model_info", None)
            if not isinstance(mi, dict):
                return None

            # Accepted optional keys:
            #   lora_update_bytes
            #   adapter_bytes
            #   lora_num_params + bytes_per_param
            dev_info = mi.get(device_name, {})
            if isinstance(dev_info, dict):
                for k in ("lora_update_bytes", "adapter_bytes"):
                    if k in dev_info:
                        return float(dev_info[k])

                if "lora_num_params" in dev_info:
                    bpp = float(dev_info.get("bytes_per_param", 4.0))
                    return float(dev_info["lora_num_params"]) * bpp

            for k in ("lora_update_bytes", "adapter_bytes"):
                if k in mi:
                    return float(mi[k])

            if "lora_num_params" in mi:
                bpp = float(mi.get("bytes_per_param", 4.0))
                return float(mi["lora_num_params"]) * bpp
        except Exception:
            return None

        return None

    # ------------------------------------------------------------------
    # Metrics/logging hook
    # ------------------------------------------------------------------
    def _write_metrics(self, phase, train_config, server_round, valid_replies):
        """Reuse parent logging, then update communication model.

        Parent _write_metrics() is responsible for CSV logging, round_stats,
        compute beta estimation, and compute time_scale calibration.

        We temporarily disable the communication term so parent compute
        calibration remains correct.
        """
        self._disable_comm_in_prediction = True
        try:
            super()._write_metrics(phase, train_config, server_round, valid_replies)
        finally:
            self._disable_comm_in_prediction = False

        if phase != "train":
            return
        if self.estimator_freeze_after_round > 0 and server_round > self.estimator_freeze_after_round:
            return

        try:
            self._update_comm_model_from_train_replies(valid_replies)
        except Exception as e:
            log(WARNING, "[LoRACastCommAware] Failed to update comm model: %s", e)

    def _update_comm_model_from_train_replies(self, valid_replies) -> None:
        """Update the client-side upload-preparation proxy without round waiting.

        Flower calls aggregation after collecting every selected reply. The
        server wall time then includes the fast client's idle wait, so it
        cannot be used as that client's communication duration.
        """

        for msg in valid_replies:
            if not msg.has_content():
                continue
            content = msg.content
            if "metrics" not in content:
                continue

            try:
                m = dict(content["metrics"].items())
            except Exception:
                continue

            device_name = self._extract_device_name_from_metrics(m)
            if device_name in ("", None):
                continue

            raw_cid = msg.metadata.src_node_id
            try:
                cid = int(raw_cid)
            except Exception:
                cid = raw_cid

            identity_key = f"{device_name}:{cid}"

            try:
                comm_time_s = float(m.get("client_upload_elapsed_time", 0.0) or 0.0)
            except Exception:
                comm_time_s = 0.0
            if comm_time_s < self.min_comm_time_s:
                continue

            arr_key = getattr(self, "arrayrecord_key", "arrays")
            arrays_rec = content.get(arr_key, None)
            payload_bytes = self._arrayrecord_nbytes(arrays_rec)

            if payload_bytes <= 0.0:
                continue

            upload_rate = payload_bytes / comm_time_s
            if upload_rate <= 0.0:
                continue

            # Update per-client and per-device-group models.
            self._ema_update(self.upload_rate_Bps, identity_key, upload_rate)
            self._ema_update(self.upload_bytes, identity_key, payload_bytes)

            self._ema_update(self.upload_rate_Bps, device_name, upload_rate)
            self._ema_update(self.upload_bytes, device_name, payload_bytes)

            log(
                INFO,
                "[LoRACastCommAware] comm model update: key=%s, bytes=%.0f, "
                "client_upload_prep_time=%.3fs, proxy_rate=%.2f B/s, ema_rate=%.2f B/s",
                identity_key,
                float(payload_bytes),
                float(comm_time_s),
                float(upload_rate),
                float(self.upload_rate_Bps.get(identity_key, 0.0)),
            )

    def _ema_update(self, table: dict[str, float], key: str, value: float) -> None:
        value = float(value)
        if key not in table or table[key] <= 0.0:
            table[key] = value
            return
        a = self.comm_ema_alpha
        table[key] = (1.0 - a) * float(table[key]) + a * value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_device_name_from_metrics(self, metrics: dict) -> str | None:
        raw_dev_name = metrics.get("device_name")
        if raw_dev_name:
            return str(raw_dev_name)

        try:
            code = metrics.get("device_code")
            # Parent device_lookup is usually {1:"RPi5", ...} on server side.
            if isinstance(getattr(self, "device_lookup", None), dict):
                return self.device_lookup.get(code) or self.device_lookup.get(int(code))
        except Exception:
            pass

        return None

    def _arrayrecord_nbytes(self, rec) -> float:
        """Estimate actual tensor payload bytes from an ArrayRecord-like object."""
        if rec is None:
            return 0.0

        arrays = self._extract_arrays(rec)
        if arrays is None:
            return 0.0

        total = 0

        try:
            if isinstance(arrays, dict):
                iterable = arrays.values()
            else:
                iterable = list(arrays)

            for v in iterable:
                try:
                    arr = np.asarray(v)
                    total += int(arr.nbytes)
                except Exception:
                    continue
        except Exception:
            return 0.0

        return float(total)

    def get_comm_debug_state(self) -> dict[str, dict[str, float]]:
        """Optional helper for debugging from a REPL/log hook."""
        return {
            "upload_rate_Bps": dict(self.upload_rate_Bps),
            "upload_bytes": dict(self.upload_bytes),
        }
