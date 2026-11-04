import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import run_trial
import export_workload


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sign(run):
    manifest = {str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in run.rglob('*') if path.is_file()}
    (run / 'checksums.json').write_text(json.dumps(manifest))


class WorkloadCsvTests(unittest.TestCase):
    def fixture(self, root, merged=False):
        prime, official = root / 'prime', root / 'official'
        calibration = official / 'reference'
        for run in (prime, calibration, official):
            run.mkdir(parents=True, exist_ok=True)
        (prime / 'state.json').write_text(json.dumps({'status': 'validated'}))
        (official / 'state.json').write_text(json.dumps({'status': 'complete'}))
        (official / 'intent.json').write_text(json.dumps({'reference': str(prime if merged else root / 'calibration')}))
        if not merged:
            (calibration / 'intent.json').write_text(json.dumps({'state_reference': str(prime)}))

        samples, copied_prime, predictions, observations = [], [], [], []
        devices = {'RPi5': ('rpi5', 1), 'jetson_orin_nx': ('jetson', 0)}
        phase_specs = ([('prime', prime, range(1, 3), 1),
                        ('official', official, range(1, 4), 2)] if merged else
                       [('prime', prime, range(1, 3), 0),
                        ('slo_calibration', calibration, range(1, 6), 1),
                        ('official', official, range(1, 4), 2)])
        for phase, run, rounds, offset in phase_specs:
            fedavg = []
            for device, (host, partition) in devices.items():
                output = run / host / 'output'
                output.mkdir(parents=True)
                events = []
                sample_reads = []
                digest = hashlib.sha256(f'{phase}:{device}'.encode()).hexdigest()
                samples.append({'phase': phase, 'device_name': device,
                                'partition_id': partition, 'partition_train_row': offset,
                                'token_ids_sha256': digest, 'input_token_count': 10 + offset,
                                'observed_rounds': ','.join(map(str, rounds))})
                for number in rounds:
                    sample_read = {'device_name': device, 'round': number,
                                   'partition_id': partition, 'partition_train_row': offset,
                                   'token_ids_sha256': digest, 'input_token_count': 10 + offset}
                    sample_reads.append(sample_read)
                    if phase == 'prime':
                        copied_prime.append(sample_read)
                    workload = {'max_samples_per_round': 1, 'selected_samples': 1,
                                'local_steps': 0, 'local_epochs': 1, 'batch_size': 1,
                                'max_seq_len_configured': 128, 'sample_offset': offset,
                                'lora_rank': 16, 'active_lora_modules': ['q_proj'],
                                'model': 'test-model', 'dataset': 'test-dataset'}
                    events.append({'kind': 'start', 'device_name': device,
                                   'device_uid': f'{device}-1', 'partition_id': partition,
                                   'round': number, 'workload': workload})
                    actual = '11.2' if phase == 'official' and number == 3 and device == 'RPi5' else '10.5'
                    fedavg.append({'round': number, 'phase': 'train', 'device_name': device,
                                   'device_uid': f'{device}-1', 'num_examples': 1,
                                   'input_tokens': 10 + offset, 'local_steps': 1,
                                   'pred_train_time_self': '10', 'actual_train_time_self': actual})
                    if phase == 'official':
                        success = actual == '10.5'
                        predictions.append({'round': number, 'device_name': device,
                                            'pred_train_time_self': '10',
                                            'actual_train_time_self': actual,
                                            'relative_error': abs(10-float(actual))/float(actual),
                                            'valid_prediction': 'True'})
                        observations.append({'round': number, 'device_name': device,
                                             'time_pass': success, 'round_energy_j': 42,
                                             'energy_violation': False})
                (output / 'training_events.jsonl').write_text(
                    ''.join(json.dumps(item) + '\n' for item in events))
                (output / 'sample_audit.jsonl').write_text(
                    ''.join(json.dumps(item) + '\n' for item in sample_reads))
            write_csv(run / 'fedavg.csv', list(fedavg[0]), fedavg)

        if merged:
            shutil.copytree(prime, calibration, dirs_exist_ok=True)
        copied = calibration / ('prime_samples.jsonl' if merged else 'state_reference/prime_samples.jsonl')
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_text(''.join(json.dumps(item) + '\n' for item in copied_prime))
        write_csv(official / 'report/tta_dataset_samples.csv', list(samples[0]), samples)
        dataset_audit = {
            'schema': 2, 'status': 'disjoint_observed', 'sample_count_cap': 1,
            'phase_sample_offsets': ({'prime':1,'official':2} if merged else
                                     {'prime': 0, 'slo_calibration': 1, 'official': 2}),
            'phase_sample_ids': {phase:[] for phase in
                                 (('prime','official') if merged else export_workload.PHASES)},
            'phase_issues': {phase: [] for phase in
                             (('prime','official') if merged else export_workload.PHASES)},
            'pairwise_overlap_counts': ({'prime_vs_official':0} if merged else
                                        {'prime_vs_slo_calibration': 0,
                                         'prime_vs_official': 0,
                                         'slo_calibration_vs_official': 0}),
        }
        (official / 'report/tta_dataset_independence.json').write_text(json.dumps(dataset_audit))
        write_csv(official / 'report/tta_prediction_audit.csv', list(predictions[0]), predictions)
        write_csv(official / 'report/tta_observations.csv', list(observations[0]), observations)
        energies=[40,41] if merged else [40,41,42,43,44]
        lock = {'calibration_round_numbers': list(range(1,len(energies)+1)),
                'calibration_energy_j': {d: energies for d in devices},
                'slo_joules': {d: max(energies)*1.1 for d in devices},
                'margin': 1.1, 'warmup_rounds': 0, 'measurement_rounds': len(energies),
                'created_at': '2026-11-02T09:00:00+09:00'}
        if merged:
            lock['slo_source']='prime'
        lock_path = calibration / 'slo_lock.json'
        lock_path.write_text(json.dumps(lock))
        summary = {'slo_lock_sha256': hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                   'slo_joules': lock['slo_joules'],
                   'criteria': {'warmup_rounds_excluded': 0}}
        (official / 'report/tta_summary.json').write_text(json.dumps(summary))
        (official / 'report/render_report.html').write_text('<html><footer id="footer"></footer></html>')
        sign(prime)
        sign(official)
        return prime, official

    def test_export_joins_actual_samples_workload_and_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prime, official = self.fixture(root)
            before = {str(p.relative_to(official)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in official.rglob('*') if p.is_file()}
            page = run_trial.publish(official, root / 'public')
            output = page.with_name('official_sample_workload.csv')
            with output.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 20)
            self.assertIn('sample_workload.csv', page.read_text())
            self.assertEqual([r['partition_train_rows'] for r in rows if r['device_name'] == 'RPi5'
                              and r['round'] == '1'], ['0', '1', '2'])
            self.assertEqual({r['sample_cap_S'] for r in rows}, {'1'})
            self.assertEqual({r['workload_log_status'] for r in rows}, {'logged'})
            self.assertEqual({r['samples_seen_in_other_phase'] for r in rows}, {'0'})
            official_rows = [r for r in rows if r['phase'] == 'official']
            self.assertEqual([r['prediction_success'] for r in official_rows].count('False'), 1)
            self.assertTrue(all(abs(float(r['energy_slo_j'])-48.4)<1e-9 for r in official_rows))
            self.assertEqual(before, {str(p.relative_to(official)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in official.rglob('*') if p.is_file()})
            prime.rename(root / 'moved_prime')
            no_prime = export_workload.build_rows(official)
            self.assertEqual([r['workload_log_status'] for r in no_prime if r['phase'] == 'prime'],
                             ['prime_run_unavailable'] * 4)
            (official / 'report/tta_dataset_samples.csv').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'signed evidence changed'):
                export_workload.build_rows(official)

    def test_two_prime_rounds_also_supply_energy_slo(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            _,official=self.fixture(root,merged=True)
            page=run_trial.publish(official,root/'public')
            with page.with_name('official_sample_workload.csv').open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),10)
            self.assertEqual({row['phase'] for row in rows},{'prime','official'})
            self.assertEqual({row['sample_window_rows'] for row in rows if row['phase']=='prime'},{'1-1'})
            self.assertEqual({row['round_energy_j'] for row in rows if row['phase']=='prime'},{'40','41'})
            self.assertEqual({row['prime_official_overlap_count'] for row in rows},{'0'})
            self.assertEqual({row['prime_slo_overlap_count'] for row in rows},{''})
            self.assertIn('Prime/SLO 보정',page.read_text())
            self.assertIn('보정 2회',run_trial.criteria_text(official))


if __name__ == '__main__':
    unittest.main()
