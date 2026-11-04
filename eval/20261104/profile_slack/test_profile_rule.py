#!/usr/bin/env python3
"""Check rounding, communication-sensitive completion, and physical caps."""
from profile_rule import choose_steps


def profile(slow_compute=100.0, slow_noncompute=10.0,
            fast_compute=50.0, fast_noncompute=10.0):
    return {
        "base_optimizer_steps": 10, "beta": 0.5, "maximum_optimizer_steps": 60,
        "devices": {
            "slow": {"base_compute_s": slow_compute, "base_noncompute_completion_s": slow_noncompute},
            "fast": {"base_compute_s": fast_compute, "base_noncompute_completion_s": fast_noncompute},
        },
    }


if __name__ == "__main__":
    assert (chosen["slow"]["H"], chosen["fast"]["H"]) == (10, 15)
    chosen = choose_steps(profile(fast_noncompute=60), sizes)
    assert (chosen["slow"]["H"], chosen["fast"]["H"]) == (10, 10)
    chosen = choose_steps(profile(slow_compute=99.9), sizes)
    assert chosen["fast"]["H"] == 14
    chosen = choose_steps(profile(slow_compute=10000), sizes)
    assert chosen["fast"]["H"] == 60
    print("PASS: original iteration rule, communication boundary, floor, and common cap")
