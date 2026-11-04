"""Deterministic, policy-constrained workload selection independent of Flower.

The cost callback must come from the deployment's original estimator. This
module cannot certify an unmeasured candidate or predict learning utility.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Candidate:
    samples: int
    epochs: int
    step_cap: int  # zero means no step cap in the original training loop
    batch: int
    sequence_cap: int
    predicted_compute_s: float
    predicted_active_energy_j: float | None = None
    original_feasible: bool = True

    def executed(self) -> tuple[int, int, float]:
        if self.samples == 0 or self.epochs == 0:
            return 0, 0, 0.0
        if min(self.samples,self.epochs,self.batch,self.sequence_cap) <= 0 or self.step_cap < 0:
            raise ValueError('invalid workload')
        batches_per_epoch = math.ceil(self.samples/self.batch)
        steps = batches_per_epoch*self.epochs
        if self.step_cap > 0:steps=min(steps,self.step_cap)
        full_passes, partial_batches = divmod(steps,batches_per_epoch)
        processed=full_passes*self.samples + min(self.samples,partial_batches*self.batch)
        return steps,processed,processed/self.samples


@dataclass(frozen=True)
class Policy:
    maximum_samples: int
    minimum_samples: int
    maximum_epochs: int
    maximum_steps: int  # zero means no separate H limit
    maximum_batch: int
    maximum_sequence_cap: int
    maximum_exposure: float | None
    prefer_fresh: bool = True


@dataclass(frozen=True)
class Decision:
    selected: Candidate | None
    action: str
    eligible_unique_samples: int
    cumulative_executed_before: int
    remaining_sample_budget_before: int | None
    intended_normal_samples: int
    actual_processed_samples: int
    actual_optimizer_steps: int
    predicted_residual_window_s: float | None


def choose_workload(*, normal: Candidate, alternatives: list[Candidate], policy: Policy,
                    eligible_unique_samples: int, cumulative_executed: int,
                    previously_visited_unique: int, compute_window_s: float) -> Decision:
    """Select maximum permissible work with fresh-first deterministic ordering.

    `alternatives` are native feasible S/H/E candidates with supplied cost
    estimates. The caller must not invent times for unsupported E values.
    """
    if eligible_unique_samples <= 0 or cumulative_executed < 0 or previously_visited_unique < 0:
        raise ValueError('invalid exposure state')
    if not math.isfinite(compute_window_s) or compute_window_s <= 0:
        raise ValueError('invalid compute window')
    if policy.maximum_exposure is not None:
        if not math.isfinite(policy.maximum_exposure) or policy.maximum_exposure < 0:
            raise ValueError('invalid X_max')
        budget=math.floor(policy.maximum_exposure*eligible_unique_samples+1e-9)
        remaining=max(0,budget-cumulative_executed)
    else:
        remaining=None
    def feasible(c: Candidate) -> bool:
        if not c.original_feasible or c.samples < policy.minimum_samples:
            return False
        if c.samples > min(policy.maximum_samples,eligible_unique_samples):return False
        if c.epochs < 1 or c.epochs > policy.maximum_epochs:return False
        if c.batch > policy.maximum_batch or c.sequence_cap > policy.maximum_sequence_cap:return False
        if policy.maximum_steps and c.executed()[0] > policy.maximum_steps:return False
        if not math.isfinite(c.predicted_compute_s) or c.predicted_compute_s > compute_window_s+1e-9:
            return False
        if remaining is not None and c.executed()[1] > remaining:return False
        return True
    # Exact NR reproduction when no user constraint binds; no alternate search.
    if feasible(normal) and policy.maximum_exposure is None and policy.maximum_epochs <= normal.epochs:
        chosen=normal
    else:
        valid=[c for c in [normal,*alternatives] if feasible(c)]
        unseen=max(0,eligible_unique_samples-previously_visited_unique)
        def ranking(c: Candidate):
            steps,processed,_=c.executed()
            fresh=min(unseen,c.samples,processed)
            energy=c.predicted_active_energy_j
            if energy is None or not math.isfinite(energy):energy=math.inf
            if policy.prefer_fresh:
                return fresh,processed,-c.predicted_compute_s,-energy,c.samples,-c.epochs,-c.step_cap
            return processed,fresh,-c.predicted_compute_s,-energy,c.samples,-c.epochs,-c.step_cap
        chosen=max(valid,key=ranking) if valid else None
    normal_work=normal.executed()[1]
    actual_steps,actual_work,_=chosen.executed() if chosen else (0,0,0.0)
    if chosen is None:action='omit_before_training'
    elif chosen==normal:action='normal'
    else:action='partial_or_bounded_epoch_quota'
    return Decision(chosen,action,eligible_unique_samples,cumulative_executed,remaining,
                    normal_work,actual_work,actual_steps,
                    max(0.0,compute_window_s-chosen.predicted_compute_s) if chosen else compute_window_s)
