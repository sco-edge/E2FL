from __future__ import annotations

import unittest

from exposure_controller import Candidate, Policy, choose_workload


def candidate(samples: int, epochs: int = 1, steps: int = 0) -> Candidate:
    return Candidate(samples,epochs,steps,1,64,float(samples*epochs))


def policy(cap: float | None, epochs: int = 1) -> Policy:
    return Policy(32,8,epochs,0,1,64,cap)


class ExposureControllerTest(unittest.TestCase):
    def decide(self, cap, total, D=128, epochs=1, normal=None):
        normal=normal or candidate(32)
        alternatives=[candidate(s) for s in range(8,33,4)]
        return choose_workload(normal=normal,alternatives=alternatives,
            policy=policy(cap,epochs),eligible_unique_samples=D,
            cumulative_executed=total,previously_visited_unique=min(D,total),
            compute_window_s=32.0)

    def test_unconstrained_matches_nr(self):
        d=self.decide(None,200)
        self.assertEqual(d.selected,candidate(32))
        self.assertEqual(d.action,'normal')

    def test_binding_partial_candidate(self):
        d=self.decide(2,232)
        self.assertEqual(d.selected.samples,24)
        self.assertEqual(d.actual_processed_samples,24)
        self.assertEqual(d.remaining_sample_budget_before,24)

    def test_under_minimum_omits_before_training(self):
        d=self.decide(2,251)
        self.assertIsNone(d.selected)
        self.assertEqual(d.action,'omit_before_training')

    def test_new_data_restores_budget(self):
        before=self.decide(2,256,D=128)
        after=self.decide(2,256,D=512)
        self.assertIsNone(before.selected)
        self.assertEqual(after.selected.samples,32)
        self.assertEqual(after.remaining_sample_budget_before,768)

    def test_step_cap_and_partial_epoch_count(self):
        c=Candidate(9,4,23,2,64,10.0)
        self.assertEqual(c.executed(),(20,36,4.0))
        c=Candidate(9,4,7,2,64,10.0)
        self.assertEqual(c.executed(),(7,13,13/9))

    def test_epoch_candidate_requires_supported_window(self):
        normal=candidate(16)
        expanded=Candidate(16,2,0,1,64,40.0)
        d=choose_workload(normal=normal,alternatives=[expanded],policy=policy(2,4),
            eligible_unique_samples=128,cumulative_executed=0,
            previously_visited_unique=0,compute_window_s=32.0)
        self.assertEqual(d.selected,normal)


if __name__=='__main__':unittest.main()
