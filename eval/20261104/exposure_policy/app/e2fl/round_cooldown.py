"""Optional barrier between completed Flower rounds for room-temperature cooling.

An external controller observes the Pi and creates a release file once its
temperature is back below the chosen starting threshold. The barrier runs
after the global checkpoint is saved and before the next round is dispatched.
"""

import json
from pathlib import Path
import time


def wait_after_round(completed_round: int, total_rounds: int, gate_dir: str,
                     timeout_s: float = 1800.0, poll_s: float = 1.0) -> None:
    if not gate_dir or completed_round >= total_rounds:
        return
    if completed_round < 1 or timeout_s <= 0 or poll_s <= 0:
        raise ValueError("invalid cooldown barrier settings")
    directory = Path(gate_dir)
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / f"request_after_round_{completed_round}.json"
    release = directory / f"release_after_round_{completed_round}.json"
    if request.exists() or release.exists():
        raise FileExistsError("cooldown barrier markers already exist")
    started = time.time()
    request.write_text(json.dumps({"completed_round": completed_round,
                                   "start_unix_s": started}) + "\n")
    deadline = time.monotonic() + timeout_s
    while not release.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"cooldown after round {completed_round} timed out")
        time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))
    released = json.loads(release.read_text())
    if released.get("completed_round") != completed_round or released.get("temperature_c") is None:
        raise ValueError("invalid cooldown release marker")
    ended = time.time()
    with (directory / "intervals.jsonl").open("a") as handle:
        handle.write(json.dumps({"completed_round": completed_round,
                                 "start_unix_s": started, "end_unix_s": ended,
                                 "duration_s": ended - started,
                                 "release_temperature_c": released["temperature_c"],
                                 "release_throttled": released.get("throttled")}) + "\n")
