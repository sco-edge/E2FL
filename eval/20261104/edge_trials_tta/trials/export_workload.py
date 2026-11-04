"""Export one readable CSV from verified TTA evidence without changing the run.

Each row is one device-round with its actually observed sample IDs. Workload settings
come from the client's start event, while outcomes come from the server CSV and
the signed TTA analysis. Prime workload is included only when its separately
signed run is still available and matches the copied prime sample evidence.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PHASES = ('prime', 'slo_calibration', 'official')
DEVICES = {'RPi5': 'rpi5', 'jetson_orin_nx': 'jetson'}
FIELDS = (
    'phase', 'round', 'device_name', 'device_uid', 'source_run',
    'partition_id', 'sample_window_rows', 'sample_count',
    'partition_train_rows', 'sample_token_ids_sha256', 'sample_input_tokens',
    'sample_tokens_per_epoch',
    'samples_seen_in_other_phase', 'samples_reused_in_other_rounds',
    'prime_slo_overlap_count', 'prime_official_overlap_count',
    'slo_official_overlap_count', 'sample_audit_status',
    'workload_log_status', 'model', 'dataset', 'sample_cap_S',
    'selected_samples', 'actual_examples', 'local_steps_limit_H',
    'local_steps_executed', 'local_epochs_E', 'batch_size_B',
    'seq_length_cap_L', 'lora_rank', 'lora_modules',
    'actual_input_tokens_all_epochs', 'pred_train_time_s',
    'actual_train_time_s', 'relative_error_pct', 'prediction_valid',
    'prediction_success', 'round_energy_j', 'energy_slo_j',
    'energy_violation',
)


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def _jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _signed(run):
    manifest = run / 'checksums.json'
    if not manifest.is_file():
        return False
    try:
        expected = _json(manifest)
        return bool(expected) and all(
            (run / name).resolve().is_relative_to(run.resolve())
            and (run / name).is_file()
            and hashlib.sha256((run / name).read_bytes()).hexdigest() == digest
            for name, digest in expected.items()
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _keyed(rows, name):
    result = {}
    for row in rows:
        if row.get('phase', 'train') != 'train':
            continue
        key = (int(row['round']), row['device_name'])
        if key in result:
            raise ValueError(f'{name}: duplicate device-round {key}')
        result[key] = row
    return result


def _training(run):
    fedavg = _keyed(_csv(run / 'fedavg.csv'), 'fedavg')
    starts = {}
    for device, host in DEVICES.items():
        for event in _jsonl(run / host / 'output/training_events.jsonl'):
            if event.get('kind') != 'start':
                continue
            key = (int(event['round']), device)
            if key in starts or event.get('device_name') != device:
                raise ValueError(f'training event mismatch: {key}')
            starts[key] = event
    if set(starts) != set(fedavg):
        raise ValueError('client workload events and server training rows differ')
    return fedavg, starts


def _prime_run(official):
    intent = official / 'reference/intent.json'
    if not intent.is_file():
        return None
    source = _json(intent).get('state_reference')
    if not source:
        return None
    candidates = (Path(source), official.parent / Path(source).name)
    for run in candidates:
        run = run.resolve()
        if not _signed(run) or _json(run / 'state.json').get('status') not in ('validated', 'complete'):
            continue
        copied = _jsonl(official / 'reference/state_reference/prime_samples.jsonl')
        actual = [entry for host in DEVICES.values() for entry in
                  _jsonl(run / host / 'output/sample_audit.jsonl')]
        identity = lambda item: (item['device_name'], item['round'],
                                 item['partition_id'], item['partition_train_row'],
                                 item['token_ids_sha256'], item['input_token_count'])
        if sorted(map(identity, copied)) == sorted(map(identity, actual)):
            return run
    return None


def build_rows(official):
    official = Path(official).resolve()
    if not _signed(official) or _json(official / 'state.json').get('status') != 'complete':
        raise ValueError('official run is incomplete or its signed evidence changed')
    paths = [official / 'report' / name for name in
             ('tta_dataset_samples.csv', 'tta_dataset_independence.json',
              'tta_prediction_audit.csv', 'tta_observations.csv', 'tta_summary.json')]
    if not all(path.is_file() for path in paths):
        protocol_path = official / 'sample_protocol.json'
        if (protocol_path.is_file() and
                _json(protocol_path).get('sample_split_policy') in
                ('three_disjoint_windows', 'two_disjoint_windows')):
            raise ValueError('three-phase run is missing a required CSV source')
        return None  # Older runs have no three-phase sample audit.
    samples_path, audit_path, prediction_path, observation_path, summary_path = paths
    dataset = _json(audit_path)
    if dataset.get('schema', 0) < 2:
        return None
    samples = _csv(samples_path)
    summary = _json(summary_path)
    lock_path = official / 'reference/slo_lock.json'
    if hashlib.sha256(lock_path.read_bytes()).hexdigest() != summary.get('slo_lock_sha256'):
        raise ValueError('SLO lock differs from the signed summary')
    lock = _json(lock_path)
    if lock['slo_joules'] != summary['slo_joules']:
        raise ValueError('signed summary and locked SLO values differ')
    merged=lock.get('slo_source')=='prime'
    phases=('prime','official') if merged else PHASES
    phase_ids = {phase: {row['token_ids_sha256'] for row in samples
                         if row['phase'] == phase} for phase in phases}
    pairwise = {left+'_vs_'+right:len(phase_ids[left] & phase_ids[right])
                for index,left in enumerate(phases) for right in phases[index+1:]}
    if pairwise != dataset.get('pairwise_overlap_counts'):
        raise ValueError('sample hashes do not agree with the signed independence audit')
    if dataset.get('phase_issues') != {phase: [] for phase in phases}:
        raise ValueError('sample audit has missing or invalid phase records')

    phase_runs = ({'prime':official/'reference','official':official} if merged else
                  {'slo_calibration': official / 'reference', 'official': official})
    calibration_source = _json(official / 'intent.json').get('reference')
    prime = None if merged else _prime_run(official)
    if not merged and prime is not None:
        phase_runs['prime'] = prime
    training = {phase: _training(run) for phase, run in phase_runs.items()}
    predictions = _keyed(_csv(prediction_path), 'prediction audit')
    observations = _keyed(_csv(observation_path), 'observations')
    if set(predictions) != set(observations) or set(observations) != set(training['official'][0]):
        raise ValueError('official training and analysis rows differ')
    energy_rounds = [int(value) for value in lock['calibration_round_numbers']]
    energy = {(number, device): values[index]
              for device, values in lock['calibration_energy_j'].items()
              for index, number in enumerate(energy_rounds)}
    if any(len(values) != len(energy_rounds)
           for values in lock['calibration_energy_j'].values()):
        raise ValueError('calibration energy and round count differ')

    observed_samples = {}
    seen = set()
    for sample in samples:
        phase = sample['phase']
        device = sample['device_name']
        if phase not in phases or device not in DEVICES:
            raise ValueError('unexpected phase or device in sample audit')
        for round_text in sample['observed_rounds'].split(','):
            round_number = int(round_text)
            key = (round_number, device)
            identity = (phase, round_number, device, sample['partition_train_row'])
            if identity in seen:
                raise ValueError(f'duplicate sample observation: {identity}')
            seen.add(identity)
            observed_samples.setdefault((phase, round_number, device), []).append(sample)
    rows = []
    for phase, round_number, device in sorted(observed_samples, key=lambda x: (phases.index(x[0]), x[1], x[2])):
        key = (round_number, device)
        group = sorted(observed_samples[(phase, round_number, device)],
                       key=lambda item: int(item['partition_train_row']))
        fedavg, starts = training.get(phase, ({}, {}))
        train = fedavg.get(key)
        start = starts.get(key)
        if phase != 'prime' and (train is None or start is None):
            raise ValueError(f'missing workload log for {phase} {key}')
        if phase == 'prime' and (prime is not None or merged) and (train is None or start is None):
            raise ValueError(f'missing prime workload log for {key}')
        workload = start['workload'] if start else {}
        offset = int(dataset['phase_sample_offsets'][phase])
        cap = int(dataset['sample_count_cap'])
        if (cap < 1 or len(group) != cap
                or {int(item['partition_train_row']) for item in group}
                != set(range(offset, offset + cap))
                or len({item['partition_id'] for item in group}) != 1):
            raise ValueError(f'incomplete sample window for {phase} {key}')
        if start and (str(start['partition_id']) != group[0]['partition_id']
                      or int(workload['sample_offset']) != offset
                      or any(not int(workload['sample_offset']) <= int(item['partition_train_row'])
                             < int(workload['sample_offset']) + int(workload['max_samples_per_round'])
                             for item in group)
                      or len(group) != int(workload['selected_samples'])):
            raise ValueError(f'client workload and observed samples disagree: {phase} {key}')
        if train and (train['device_uid'] != start['device_uid']
                      or int(train['num_examples']) != int(workload['selected_samples'])):
            raise ValueError(f'server and client workload disagree: {key}')
        prediction = predictions.get(key) if phase == 'official' else None
        observation = observations.get(key) if phase == 'official' else None
        if prediction and train and (prediction['pred_train_time_self'] != train['pred_train_time_self']
                                     or prediction['actual_train_time_self'] != train['actual_train_time_self']):
            raise ValueError(f'prediction audit and server training disagree: {key}')
        if prediction and prediction['valid_prediction'].lower() == 'true':
            succeeds = float(prediction['relative_error']) <= 0.10
            if succeeds != (observation['time_pass'].lower() == 'true'):
                raise ValueError(f'prediction result and relative error disagree: {key}')
        if observation and ((float(observation['round_energy_j']) > float(lock['slo_joules'][device]))
                            != (observation['energy_violation'].lower() == 'true')):
            raise ValueError(f'energy result and locked SLO disagree: {key}')
        hashes_elsewhere = set().union(*(phase_ids[p] for p in phases if p != phase))
        row = {field: '' for field in FIELDS}
        row.update({
            'phase': phase, 'round': round_number, 'device_name': device,
            'device_uid': start['device_uid'] if start else '',
            'source_run': (Path(calibration_source).name if phase in ('slo_calibration','prime')
                           and merged and calibration_source else
                           Path(calibration_source).name if phase == 'slo_calibration' and calibration_source
                           else phase_runs[phase].name if phase in phase_runs else ''),
            'partition_id': group[0]['partition_id'],
            'sample_window_rows': f'{offset}-{offset + cap - 1}',
            'sample_count': len(group),
            'partition_train_rows': '|'.join(item['partition_train_row'] for item in group),
            'sample_token_ids_sha256': '|'.join(item['token_ids_sha256'] for item in group),
            'sample_input_tokens': '|'.join(item['input_token_count'] for item in group),
            'sample_tokens_per_epoch': sum(int(item['input_token_count']) for item in group),
            'samples_seen_in_other_phase': sum(item['token_ids_sha256'] in hashes_elsewhere
                                               for item in group),
            'samples_reused_in_other_rounds': sum(len(item['observed_rounds'].split(',')) > 1
                                                  for item in group),
            'prime_slo_overlap_count': pairwise.get('prime_vs_slo_calibration',''),
            'prime_official_overlap_count': pairwise['prime_vs_official'],
            'slo_official_overlap_count': pairwise.get('slo_calibration_vs_official',''),
            'sample_audit_status': dataset['status'],
            'workload_log_status': 'logged' if start else 'prime_run_unavailable',
        })
        if start:
            row.update({
                'model': workload['model'], 'dataset': workload['dataset'],
                'sample_cap_S': workload['max_samples_per_round'],
                'selected_samples': workload['selected_samples'],
                'actual_examples': train['num_examples'],
                'local_steps_limit_H': workload['local_steps'],
                'local_steps_executed': train['local_steps'],
                'local_epochs_E': workload['local_epochs'],
                'batch_size_B': workload['batch_size'],
                'seq_length_cap_L': workload['max_seq_len_configured'],
                'lora_rank': workload['lora_rank'],
                'lora_modules': '|'.join(workload['active_lora_modules']),
                'actual_input_tokens_all_epochs': train['input_tokens'],
                'actual_train_time_s': train['actual_train_time_self'],
            })
        if phase == ('prime' if merged else 'slo_calibration'):
            row['round_energy_j'] = energy[key]
            row['energy_slo_j'] = lock['slo_joules'][device]
        elif phase == 'official':
            row.update({
                'pred_train_time_s': prediction['pred_train_time_self'],
                'relative_error_pct': 100 * float(prediction['relative_error']),
                'prediction_valid': prediction['valid_prediction'],
                'prediction_success': (observation['time_pass']
                                       if prediction['valid_prediction'].lower() == 'true' else ''),
                'round_energy_j': observation['round_energy_j'],
                'energy_slo_j': lock['slo_joules'][device],
                'energy_violation': observation['energy_violation'],
            })
        rows.append(row)
    expected = {(phase, int(r), device) for phase, (fedavg, _) in training.items()
                for r, device in fedavg}
    actual = {(row['phase'], row['round'], row['device_name']) for row in rows}
    if not expected <= actual or not set(predictions) <= {(r, d) for p, r, d in actual if p == 'official'}:
        raise ValueError('sample observations do not cover all logged device-rounds')
    return rows


def export(official, path):
    rows = build_rows(official)
    if rows is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path
