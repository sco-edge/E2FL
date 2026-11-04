import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_trial as viewer


class ResultPageTests(unittest.TestCase):
    def fixture(self, root):
        run = root / 'trial'
        (run / 'report').mkdir(parents=True)
        (run / 'reference').mkdir()
        (run / 'state.json').write_text(json.dumps({'status': 'complete', 'start': 1_000_000,
                                                   'transport': 'direct'}))
        stats = {'complete': True, 'time_pass': True, 'energy_pass': True,
                 'time_prediction_accuracy_pct': 100.0,
                 'observations': 3, 'expected_observations': 3}
        summary = {'pass': True,'audit_status':'verified',
                   'prediction_audit_valid':True,'slo_audit_valid':True,
                   'dataset_independence_pass':True,
                   'overall': {**stats, 'observations': 6, 'expected_observations': 6,
                               'energy_violation_rate_pct': 0},
                   'per_device': {'RPi5': stats, 'jetson_orin_nx': stats},
                   'slo_joules': {'RPi5': 400, 'jetson_orin_nx': 600},
                   'criteria': {'time_accuracy_target': .9, 'energy_violation_target': .1,
                                'accuracy_metric': 'within_error_limit',
                                'warmup_rounds_excluded': 0,
                                'energy_definition': 'Measured interval.'},
                   'onsite_runtime': {'elapsed_s': 364, 'target_s': 600, 'within_target': True},
                   'slo_lock_sha256': ''}
        lock = {'margin': 1.1, 'warmup_rounds': 0, 'measurement_rounds': 5,
                'created_at': '2026-10-10T09:15:00+09:00',
                'calibration_energy_j': {'RPi5': [300, 310, 320, 330, 400/1.1],
                                         'jetson_orin_nx': [400, 450, 500, 530, 600/1.1]}}
        lock_path = run / 'reference/slo_lock.json'
        lock_path.write_text(json.dumps(lock))
        summary['slo_lock_sha256'] = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        (run / 'report/tta_summary.json').write_text(json.dumps(summary))
        with (run / 'report/tta_observations.csv').open('w', newline='') as f:
            fields = ['round','device_name','pred_train_time_s','actual_train_time_s',
                      'relative_error','round_energy_j','energy_slo_j','time_pass','energy_violation']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in range(1, 4):
                for device in ('RPi5', 'jetson_orin_nx'):
                    writer.writerow(dict(round=r, device_name=device, pred_train_time_s=20,
                                         actual_train_time_s=21, relative_error=1/21,
                                         round_energy_j=300, energy_slo_j=400,
                                         time_pass=True,
                                         energy_violation=False))
        hashes = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in run.rglob('*') if p.is_file()}
        (run / 'checksums.json').write_text(json.dumps(hashes))
        return run

    def test_verified_page_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = self.fixture(root)
            before = sorted(str(p.relative_to(run)) for p in run.rglob('*') if p.is_file())
            result = viewer.publish(run, root / 'public')
            content = result.read_text()
            self.assertIn('3라운드 측정 통과', content)
            self.assertIn('6분 04초', content)
            self.assertIn('증거 확인: 4개 파일 해시 일치', content)
            self.assertIn('개별 기준 이내', content)
            self.assertIn('상대오차 10% 이하', content)
            self.assertIn('에너지 상한을 정한 이유', content)
            self.assertIn('절감 효과를 입증하지 않습니다', content)
            self.assertIn('400.00 J', viewer.criteria_text(run))
            self.assertTrue((root / 'public/latest_criteria.txt').is_file())
            self.assertTrue((root / 'public/latest.html').is_file())
            self.assertEqual(before, sorted(str(p.relative_to(run)) for p in run.rglob('*') if p.is_file()))
            (run / 'report/tta_observations.csv').write_text('tampered')
            self.assertIn('결과 확인 필요', viewer.page(run))
            self.assertIn('증거 파일 변경', viewer.page(run))

    def test_invalid_run_escapes_diagnosis(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'failed'
            run.mkdir()
            (run / 'diagnosis.json').write_text(json.dumps({'error': '<script>alert(1)</script>',
                                                            'suggestions': ['check <device>']}))
            content = viewer.page(run)
            self.assertIn('유효한 시험 결과가 없습니다', content)
            self.assertIn('&lt;script&gt;', content)
            self.assertNotIn('<script>', content)

    def test_run_command_publishes_latest_attempt_for_both_modes(self):
        class FakeProcess:
            stdout = ['Run: /tmp/first_invalid\n', 'INVALID: retry\n',
                      'Run: /tmp/last_attempt\n', 'OVERALL: PASS\n']
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def wait(self): return 0
        for mode, folder in (('reuse','edge_trials/harness'), ('standard','tta_harness')):
            with self.subTest(mode=mode), patch.object(viewer.subprocess, 'Popen', return_value=FakeProcess()) as popen, \
                 patch.object(viewer, 'publish', return_value=Path('/tmp/result.html')) as publish:
                with patch.object(Path, 'is_file', return_value=True):
                    argv=['run','--mode',mode,'--reference','/tmp/reference','--no-open']
                    if mode=='standard': argv.append('--allow-legacy')
                    self.assertEqual(viewer.main(argv),0)
                self.assertIn(folder, popen.call_args.args[0][1])
                publish.assert_called_once_with(Path('/tmp/last_attempt'))


if __name__ == '__main__':
    unittest.main()
