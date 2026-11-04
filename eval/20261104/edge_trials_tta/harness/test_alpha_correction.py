"""Behavioral checks for the correction applied only to disposable TTA snapshots."""
import math
import unittest
from pathlib import Path


PATCH = Path(__file__).resolve().parents[2]/'compatibility/patches/online_time_scale_correction.patch'


def patched_method_class():
    lines = PATCH.read_text().splitlines()
    start = lines.index('+    def _update_time_scale(self, identity_key: str, predicted: float, observed: float) -> None:')
    added = []
    for line in lines[start:]:
        if not line.startswith('+'):
            break
        added.append(line[1:])
    namespace = {'math': math}
    exec('class Dummy:\n'+'\n'.join(added)+'\n', namespace)
    cls = namespace['Dummy']
    cls._split_identity_key = lambda self, key: (key.split(':', 1)[0], '')
    return cls


class AlphaCorrectionTests(unittest.TestCase):
    def test_residual_ema_uses_existing_scale_and_preexisting_group_fallback(self):
        strategy = patched_method_class()()
        strategy.time_scale = {'RPi5': 1.2}
        strategy.calib_alpha = 0.3
        strategy.calib_clip = (0.5, 2.0)
        strategy._update_time_scale('RPi5:pi-5', 12.0, 15.0)
        self.assertAlmostEqual(strategy.time_scale['RPi5:pi-5'], 1.29)
        self.assertEqual(strategy.time_scale['RPi5'], 1.2)
        strategy._update_time_scale('RPi5:pi-5', 20.0, 20.0)
        self.assertAlmostEqual(strategy.time_scale['RPi5:pi-5'], 1.29)

    def test_invalid_observations_do_not_change_scale_and_residual_is_clipped(self):
        strategy = patched_method_class()()
        strategy.time_scale = {}
        strategy.calib_alpha = 0.3
        strategy.calib_clip = (0.5, 2.0)
        strategy._update_time_scale('RPi5', 10.0, 100.0)
        self.assertAlmostEqual(strategy.time_scale['RPi5'], 1.3)
        for predicted, observed in ((0, 5), (10, 0), (math.inf, 10), (10, math.nan)):
            strategy._update_time_scale('RPi5', predicted, observed)
            self.assertAlmostEqual(strategy.time_scale['RPi5'], 1.3)


if __name__ == '__main__':
    unittest.main()
