"""Isolated, prediction-only participation selector. No Flower dependency.

This module never changes the original LoRACast estimator or candidate search.
Candidates must already have passed the original time/memory feasibility checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


@dataclass(frozen=True)
class CompletedRound:
    round_id: int
    slow_work: float
    fast_work: float


@dataclass(frozen=True)
class Candidate:
    samples: int
    steps: int  # H=0 denotes no positive step cap in the original implementation.
    predicted_work: float
    predicted_seconds: float
    original_feasible: bool


@dataclass(frozen=True)
class Decision:
    mode: str
    candidate: Candidate | None
    retained_rounds: tuple[int, ...]
    retained_slow_work: float
    retained_fast_work: float
    prior_share: float | None
    projected_share: float | None
    fast_work_bound: float | None
    reason: str


def projected_fast_bound(a_s: float, a_f: float, x_s: float, theta: float) -> float:
    if not 0 < theta < 1:
        raise ValueError("effective participation floor must be in (0, 1)")
    if min(a_s, a_f, x_s) < 0 or not all(map(isfinite, (a_s, a_f, x_s, theta))):
        raise ValueError("history and prediction must be finite nonnegative values")
    return ((1 - theta) / theta) * (a_s + x_s) - a_f


def choose_quota(
    *,
    next_round: int,
    completed: Sequence[CompletedRound],
    window_w: int,
    theta_eff: float,
    expected_slow_work: float,
    normal: Candidate,
    alternatives: Sequence[Candidate],
    warmup_rounds: int = 2,
    previous_regulation_round: int | None = None,
) -> Decision:
    """Apply original trigger/cooldown to a nonzero, discrete fast quota.

    `completed` contains actual work from completed rounds only. Callers must
    supply candidates already checked by the original allocator's feasibility
    logic; an unknown predicted cost is never treated as feasible.
    """
    if window_w < 1 or next_round < 1:
        raise ValueError("invalid round/window")
    if not 0 < theta_eff < 1 or not isfinite(theta_eff):
        raise ValueError("invalid effective floor")
    if expected_slow_work < 0 or not isfinite(expected_slow_work):
        raise ValueError("invalid expected slow work")
    if any(r.round_id >= next_round or min(r.slow_work, r.fast_work) < 0 for r in completed):
        raise ValueError("future or negative completed work")
    if len({r.round_id for r in completed}) != len(completed):
        raise ValueError("duplicate round")
    kept = tuple(sorted(completed, key=lambda r: r.round_id)[-max(0, window_w - 1):]) if window_w > 1 else ()
    a_s = sum(r.slow_work for r in kept)
    a_f = sum(r.fast_work for r in kept)
    total = a_s + a_f
    prior_share = a_s / total if total > 0 else None
    base = dict(retained_rounds=tuple(r.round_id for r in kept), retained_slow_work=a_s,
                retained_fast_work=a_f, prior_share=prior_share)
    if next_round <= warmup_rounds or prior_share is None:
        return Decision("NORMAL", normal, **base, projected_share=None, fast_work_bound=None,
                        reason="warmup or no completed-work history")
    if previous_regulation_round == next_round - 1:
        return Decision("NORMAL", normal, **base, projected_share=None, fast_work_bound=None,
                        reason="one forced normal round after regulation")
    if prior_share >= theta_eff:
        return Decision("NORMAL", normal, **base, projected_share=None, fast_work_bound=None,
                        reason="prior retained share meets effective floor")
    bound = projected_fast_bound(a_s, a_f, expected_slow_work, theta_eff)
    eligible = [c for c in alternatives if c.samples > 0 and c.steps > 0 and
                c.original_feasible and isfinite(c.predicted_seconds) and c.predicted_seconds > 0 and
                isfinite(c.predicted_work) and 0 < c.predicted_work <= min(bound, normal.predicted_work) + 1e-9 and
                c.predicted_seconds <= normal.predicted_seconds + 1e-9]
    if not eligible:
        return Decision("OMIT_FALLBACK", None, **base, projected_share=None,
                        fast_work_bound=bound, reason="no positive candidate meets floor and native feasibility")
    # Equal predicted work needs no larger sample cap or data preparation.
    chosen = max(eligible, key=lambda c: (c.predicted_work, -c.samples, -c.steps))
    denominator = a_s + a_f + expected_slow_work + chosen.predicted_work
    share = (a_s + expected_slow_work) / denominator if denominator > 0 else None
    return Decision("THROTTLED_BOTH", chosen, **base, projected_share=share,
                    fast_work_bound=bound, reason="largest positive native-feasible candidate within share bound")
