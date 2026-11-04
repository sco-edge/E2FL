"""Read-only Flower integration check for the isolated Nano comparison."""
from pathlib import Path
from types import SimpleNamespace
import json

from flwr.app import ConfigRecord
import server_app_nano_reference as server


root = Path(__file__).resolve().parent
common = {
    "fraction-train": 1.0,
    "min-clients": 2,
    "static-allocation-after-round": 0,
    "enable-rpi-only-rounds": True,
    "use-token-ratio": True,
    "participation-guard-mode": "per-client",
    "participation-fairness-rho": 0.6,
    "aggregation-mode": "processed_samples",
    "partition-manifest-path": str(root / "manifests/alpaca-cleaned/v2/loracast_strong.json"),
    "experiment-output-root": str(root.parent / "preflight_results"),
    "reference-nano-samples": 320,
    "reference-nano-steps": 320,
    "reference-nano-correction-max-samples": 320,
    "reference-nano-correction-max-steps": 320,
    "reference-nano-correction-max-seconds": 240.0,
    "model": "hf:Qwen/Qwen2.5-0.5B-Instruct",
    "use-comm-prediction": False,
}
results = {}
for name in ("NanoReferenceNR", "NanoReferenceOmit", "NanoReferenceThrottle"):
    config = dict(common, strategy=name, **{"experiment-id": "nano-import-" + name.lower()})
    subject = server.create_strategy(SimpleNamespace(run_config=config))
    assert subject.rebalancing_enabled is False
    assert subject.participation_action == {"NanoReferenceNR": "none", "NanoReferenceOmit": "omit", "NanoReferenceThrottle": "throttle"}[name]
    train_config = ConfigRecord({
        "batch_size": 1, "local_epochs": 1, "max_samples_per_round": 960,
        "max_seq_len": 64, "window-W": 3, "policy-warmup-rounds": 2,
        "rpi-normal-quota": 320, "rpi-sample-step": 2,
        "rpi-correction-max-quota": 320, "max-correction-time-s": 240.0,
        "jetson-base-quota": 480, "jetson-max-quota": 480,
        "jetson-max-steps": 480, "jetson-sample-step": 4,
        "jetson-step-increment": 1, "jetson-target-margin": 0.95,
    })
    train_config.update(subject.nano_config)
    selected = subject.update_train_config(1, train_config)
    assert selected["round_mode"] == "BOTH"
    assert selected["max_samples_per_round_jetson_orin_nano_super"] == 320
    assert selected["local_steps_jetson_orin_nano_super"] == 320
    assert selected["max_samples_per_round_jetson_orin_nx"] == 480
    results[name] = {
        "mode": selected["round_mode"],
        "nano_samples": selected["max_samples_per_round_jetson_orin_nano_super"],
        "nano_steps": selected["local_steps_jetson_orin_nano_super"],
        "nx_samples": selected["max_samples_per_round_jetson_orin_nx"],
        "nx_steps": selected["local_steps_jetson_orin_nx"],
    }
assert len({tuple(sorted(v.items())) for v in results.values()}) == 1
print(json.dumps({"status": "passed", "modes": results}, indent=2))
