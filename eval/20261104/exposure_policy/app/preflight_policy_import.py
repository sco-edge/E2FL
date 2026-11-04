"""Server-side no-device import and round-1 request validation."""
import json
import tomli as tomllib
from pathlib import Path
from types import SimpleNamespace

from flwr.app import ConfigRecord
import server_app_nano_reference as server

root = Path(__file__).resolve().parent
defaults = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
observed = {}
for code, name in (("F", "NanoPolicyFixed"), ("A", "NanoPolicyFrozen"),
                   ("B", "NanoPolicyLastRate"), ("C", "NanoPolicyOnline")):
    override = tomllib.loads(next((root / "config/nano-policy-pilot").glob(f"*-{code.lower()}-r12-*.toml")).read_text())
    run_config = {**defaults, **override, "experiment-id": f"dry-import-{code.lower()}",
                  "experiment-output-root": str(root / "dry_import")}
    strategy = server.create_strategy(SimpleNamespace(run_config=run_config))
    train = ConfigRecord({k: v for k, v in run_config.items()
                          if isinstance(v, (int, float, bool, str))})
    request = strategy.update_train_config(1, train)
    pair = (int(request["max_samples_per_round_jetson_orin_nano_super"]),
            int(request["local_steps_jetson_orin_nano_super"]),
            int(request["max_samples_per_round_jetson_orin_nx"]),
            int(request["local_steps_jetson_orin_nx"]))
    if pair != (320, 320, 360, 0):
        raise AssertionError((name, pair))
    observed[code] = {"class": type(strategy).__name__, "round1_requested_s_h": pair}
print(json.dumps({"status": "passed", "policies": observed}, indent=2))
