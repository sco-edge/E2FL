"""Opt-in ServerApp entrypoint for the three Nano+NX experimental variants.

Load this module only from a separately approved isolated Flower package. It
reuses the historical ServerApp main/checkpoint/evaluation code but supplies a
Nano/NX strategy factory. The original repository module is not edited.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from e2fl import server_app_loracast as original_server
from e2fl.utils import cfg
from nano_reference_strategy import NanoReferenceLoRACast
from nano_policy_variants import (NanoPolicyFixed, NanoPolicyFrozen,
                                  NanoPolicyLastRate, NanoPolicyOnline,
                                  NanoUniform, NanoPyramidBounded)
from nano_final_exposure import NanoFinalExposure


def _bool(value):
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    raise ValueError(f"invalid Boolean run config value: {value!r}")


_original_factory = original_server.create_strategy
_strategies = {"NanoReferenceNR": "none", "NanoReferenceOmit": "omit",
               "NanoReferenceThrottle": "throttle"}
_policy_strategies = {
    "NanoPolicyFixed": NanoPolicyFixed,
    "NanoPolicyFrozen": NanoPolicyFrozen,
    "NanoPolicyLastRate": NanoPolicyLastRate,
    "NanoPolicyOnline": NanoPolicyOnline,
    "NanoUniform": NanoUniform,
    "NanoPyramidBounded": NanoPyramidBounded,
    "NanoFinalExposure": NanoFinalExposure,
}
original_server.device_lookup = {1: "jetson_orin_nano_super", 2: "jetson_orin_nx"}


def create_strategy(context):
    name = str(cfg(context, "strategy", ""))
    if name not in _strategies and name not in _policy_strategies:
        return _original_factory(context)
    if int(cfg(context, "min-clients", 0)) != 2:
        raise ValueError("Nano reference comparison requires exactly two clients")
    if int(cfg(context, "static-allocation-after-round", 0)) != 0:
        raise ValueError("static allocation is incompatible with this comparison")
    if not (_bool(cfg(context, "enable-rpi-only-rounds", True)) and
            _bool(cfg(context, "use-token-ratio", True))):
        raise ValueError("Nano regulation requires protected-message filtering and token share")
    if str(cfg(context, "participation-guard-mode", "per-client")) != "per-client":
        raise ValueError("Nano variant supports only the per-client guard")

    experiment_id = str(cfg(context, "experiment-id", "") or "").strip()
    if not experiment_id or not all(c.isalnum() or c in "._-" for c in experiment_id):
        raise ValueError("a new safe experiment-id is required")
    experiment_root = os.path.abspath(os.path.expanduser(str(
        cfg(context, "experiment-output-root", "") or
        os.path.join(original_server._LOCAL_PATH, "eval", "experiments"))))
    manifest_path = str(cfg(context, "partition-manifest-path", "") or "")
    if not manifest_path:
        raise ValueError("a frozen two-client partition manifest is required")
    source_manifest = Path(manifest_path)
    if not source_manifest.is_absolute():
        source_manifest = Path(original_server._LOCAL_PATH) / source_manifest
    data = json.loads(source_manifest.read_text())
    if int(data.get("num_clients", 0)) != 2:
        raise ValueError("partition manifest must contain exactly two clients")
    profiles = context.run_config.get("device-profiles-path")
    if profiles and not os.path.isabs(str(profiles)):
        profiles = os.path.join(original_server._LOCAL_PATH, str(profiles))

    nano_keys = ("reference-nano-samples", "reference-nano-steps",
                 "reference-nano-correction-max-samples",
                 "reference-nano-correction-max-steps",
                 "reference-nano-correction-max-seconds")
    nano_config = {key: context.run_config[key] for key in nano_keys}
    klass = _policy_strategies.get(name, NanoReferenceLoRACast)
    variant_kwargs = ({"participation_action": _strategies[name]}
                      if name in _strategies else {})
    if name == "NanoPyramidBounded":
        variant_kwargs.update(base_quota=320, min_quota=320,
                              max_scale=600 / 320, speed_window=5,
                              scale_epochs_after_quota=False)
    if name == "NanoFinalExposure":
        if str(cfg(context, "sample-selection-mode", "prefix")) != "exposure_aware":
            raise ValueError("NanoFinalExposure requires measured sample visits")
        if str(cfg(context, "aggregation-mode", "processed_samples")) != "processed_samples":
            raise ValueError("NanoFinalExposure requires processed-sample aggregation")
        exposure_text = str(cfg(context, "exposure-max", "none")).strip().lower()
        variant_kwargs.update(
            exposure_max=None if exposure_text in ("none", "disabled", "") else float(exposure_text),
            nx_eligible_unique=int(cfg(context, "nx-eligible-unique", 0)),
            nx_minimum_samples=int(cfg(context, "nx-minimum-samples", 0)),
            nx_maximum_samples=int(cfg(context, "nx-maximum-samples", 0)),
            nx_maximum_epochs=int(cfg(context, "nx-maximum-epochs", 1)),
        )
    is_baseline = name in {"NanoUniform", "NanoPyramidBounded"}
    if not is_baseline:
        variant_kwargs["nano_config"] = nano_config
        variant_kwargs.update(
            comm_ema_alpha=float(cfg(context, "comm-ema-alpha", .3)),
            comm_safety_factor=float(cfg(context, "comm-safety-factor", 1.0)),
            use_comm_prediction=_bool(cfg(context, "use-comm-prediction", True)),
            static_allocation_after_round=0,
        )
    return klass(
        **variant_kwargs,
        fraction_train=float(context.run_config["fraction-train"]),
        fraction_evaluate=1.0, min_available_nodes=2,
        log_dir=os.path.join(experiment_root, experiment_id, "server"),
        model_name=str(context.run_config.get("model", "unknown_model")),
        latte=False, model_info=None,
        device_lookup=original_server.device_lookup,
        ROOT=original_server._LOCAL_PATH,
        device_profiles_path=profiles,
        tokens_per_example=float(cfg(context, "tokens-per-example", 256)),
        enable_rpi_only_rounds=not is_baseline,
        use_token_ratio=not is_baseline,
        enable_deadline_guard=_bool(cfg(context, "enable-deadline-guard", False)),
        participation_guard_mode="per-client",
        participation_fairness_rho=float(cfg(context, "participation-fairness-rho", .3)),
        aggregation_mode=str(cfg(context, "aggregation-mode", "processed_samples")),
        # The inherited guard is RPi-specific. Disable only that guard here;
        # Nano omission/throttling is selected by NanoReferenceLoRACast above.
        experiment_id=experiment_id, rebalancing_enabled=False,
        expected_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        expected_manifest_client_count=2,
        estimator_freeze_after_round=int(cfg(context, "estimator-freeze-after-round", 0)),
        residual_calibration_enabled=_bool(cfg(context, "residual-calibration-enabled", False)),
    )


original_server.create_strategy = create_strategy
app = original_server.app
