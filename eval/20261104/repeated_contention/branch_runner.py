#!/usr/bin/env python3
"""Isolated corrected-floor Nano+NX Strong RTX FL with native Flower strategy.

The strategy receives only this policy's previous completed work and the
predeclared normal-condition edge *local-compute* projection. The Flower
configure/aggregate hooks make the decisions and update native state. This is
an RTX learning experiment with simulated edge feedback, not an edge run.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL_V2 = HERE.parent / 'training_cost_projection' / 'nano_full100_lower_floor_v2'
V2 = Path(os.environ.get('LORACAST_NANO_FLOOR256_V2',
                         str(LOCAL_V2 if LOCAL_V2.is_dir() else HERE.parent / 'lower_floor_v2')))
NANO_PROTOCOL = Path(os.environ.get('LORACAST_NANO_PROTOCOL',
    str(HERE.parent / 'training_cost_projection' / 'edge_protocol'
        if (HERE.parent / 'training_cost_projection' / 'edge_protocol').is_dir()
        else HERE.parent)))
sys.path.insert(0, str(NANO_PROTOCOL))
from edge_profile_oracle import FROZEN, digest, project
from nano_qwen_rtx_runner import (CHECKPOINT_ROUNDS, CORE_DIR, EXPECTED_INITIAL_SHA,
                                  EXPECTED_PARTITION_SHA, EXPECTED_STRONG_SHA,
                                  PARTITIONS, SOURCE, append_jsonl, load_jsonl)


MANIFEST = HERE / 'experiment_manifest.json'
PARENT_MANIFEST = V2 / 'nano_lower_floor_experiment_manifest_v2.json'
HASHES = V2 / 'nano_adaptive_native_source_hashes_v1.json'
STRESS_SOURCE = V2 / 'source_stress_manifest_snapshot.json'
OUT_ROOT = HERE / 'runs'
NANO = 'jetson_orin_nano_super'
NX = 'jetson_orin_nx'


class TwoClientGrid:
    def get_node_ids(self):
        return [1, 2]


def _unused_v2_prelaunch_check(run_id: str, *, require_launch: bool) -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text())
    matches = [r for r in manifest['runs'] if r['run_id'] == run_id]
    if len(matches) != 1 or matches[0]['policy'] != 'LoRACast-Exposure':
        raise ValueError('run ID is absent, duplicated, or outside the corrected LoRACast policy')
    if manifest['schema_version'] != 2 or manifest['run_common']['rounds'] != 100:
        raise ValueError('corrected-floor manifest schema or horizon changed')
    policy_config = manifest['controller']['native_train_config']['LoRACast-Exposure']
    variant = manifest['variant_provenance']
    if (int(policy_config['jetson-base-quota']) != 256 or
            int(policy_config['jetson-sample-step']) != 4 or
            int(policy_config['jetson-max-quota']) != 600 or
            float(policy_config['jetson-target-margin']) != .95 or
            int(policy_config['local-epochs']) != 1 or
            variant['old_value'] != 360 or variant['new_value'] != 256):
        raise ValueError('frozen corrected-floor policy changed')
    schedule = manifest['stress_schedule']
    stress_source = json.loads(STRESS_SOURCE.read_text())
    factors = schedule['factor_rounds']
    if (digest(STRESS_SOURCE) != schedule['source_sha256'] or
            factors != stress_source['schedule'] or
            schedule['normal_rounds'] != [1, 92] or
            [int(row['extension_round']) for row in factors] != list(range(93, 101)) or
            [row['phase'] for row in factors] != ['contention'] * 4 + ['recovery'] * 4 or
            [int(row['fig9_reference_round']) for row in factors] != list(range(5, 13)) or
            any(not math.isfinite(float(row['nano_compute_seconds_per_token_factor'])) or
                float(row['nano_compute_seconds_per_token_factor']) <= 0 for row in factors)):
        raise ValueError('frozen Fig. 9 slowdown/recovery schedule changed')
    static_manifest = Path(os.environ.get('LORACAST_NANO_STATIC_MANIFEST',
                                          str(NANO_PROTOCOL / 'nano_experiment_manifest.json')))
    for path, expected in ((SOURCE / 'initial_adapter_round0.pt', EXPECTED_INITIAL_SHA),
                           (SOURCE / 'loracast_strong.json', EXPECTED_STRONG_SHA),
                           (PARTITIONS, EXPECTED_PARTITION_SHA),
                           (static_manifest, manifest['static_manifest_sha256']),
                           (CORE_DIR / 'run_qwen_training.py', manifest['rtx_core_runner_sha256']),
                           (NANO_PROTOCOL / 'nano_qwen_rtx_run_adaptive_training.py', manifest['static_nano_runner_sha256']),
                           (FROZEN, manifest['frozen_edge_profile_sha256']),
                           (HASHES, manifest['adaptive_native_source_hashes_sha256']),
                           (NANO_PROTOCOL / 'native_parity_result_seed43.json',
                            manifest['native_past_log_parity_sha256'])):
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'frozen input missing or hash changed: {path}')
    if not (CORE_DIR / 'run_qwen_training.py').is_file():
        raise FileNotFoundError(CORE_DIR / 'run_qwen_training.py')
    original_root = Path(manifest['controller']['original_source_root'])
    for relative, expected in json.loads(HASHES.read_text())['files'].items():
        path = original_root / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'original Flower source missing or changed: {path}')
    parity = json.loads((NANO_PROTOCOL / 'native_parity_result_seed43.json').read_text())
    if not all(parity['comparison'][p]['all_match'] and parity['comparison'][p]['rounds'] == 12
               for p in ('P', 'C')):
        raise ValueError('past-log native source decision parity is not complete')
    if require_launch and (not manifest['adaptive_launch_allowed'] or
                           os.environ.get('LORACAST_NANO_FLOOR256_RTX_APPROVED') != '1'):
        raise RuntimeError('adaptive RTX launch gate is closed')
    return manifest, matches[0]


def branch_manifest() -> dict:
    branch = json.loads(MANIFEST.read_text())
    if (branch['schema_version'] != 1 or branch['source_round'] != 49 or
            branch['first_branch_round'] != 50 or branch['last_round'] != 100 or
            branch['branch_run_id'] == branch['source_run_id']):
        raise ValueError('branch identity or horizon changed')
    if digest(PARENT_MANIFEST) != branch['source_manifest_sha256']:
        raise ValueError('parent floor256 manifest hash changed')
    if digest(V2 / 'nano_lower_floor_rtx_runner_v2.py') != branch['source_runner_sha256']:
        raise ValueError('parent floor256 run_adaptive_training hash changed')
    if digest(STRESS_SOURCE) != branch['source_stress_snapshot_sha256']:
        raise ValueError('measured factor source hash changed')
    parent = json.loads(PARENT_MANIFEST.read_text())
    measured = parent['stress_schedule']['factor_rounds']
    factors = branch['factor_rounds']
    if len(factors) != 51 or [x['round'] for x in factors] != list(range(50, 101)):
        raise ValueError('branch schedule must explicitly cover R50–R100')
    for item in factors:
        offset = (item['round'] - 50) % 12
        phase = 'normal' if offset < 4 else 'contention' if offset < 8 else 'recovery'
        measured_row = None if offset < 4 else measured[offset - 4]
        expected_factor = 1.0 if measured_row is None else measured_row['nano_compute_seconds_per_token_factor']
        expected_reference = None if measured_row is None else measured_row['fig9_reference_round']
        if (item['phase'] != phase or item['fig9_reference_round'] != expected_reference or
                item['nano_compute_seconds_per_token_factor'] != expected_factor or
                not math.isfinite(float(expected_factor)) or float(expected_factor) <= 0):
            raise ValueError(f'R{item["round"]} schedule differs from frozen Fig. 9 factors')
    if parent['controller']['native_train_config']['LoRACast-Exposure']['jetson-base-quota'] != 256:
        raise ValueError('parent corrected floor changed')
    return branch


def prelaunch_check(run_id: str, *, require_launch: bool) -> tuple[dict, dict]:
    branch = branch_manifest()
    if run_id != branch['branch_run_id']:
        raise ValueError('unknown or source run ID')
    sys.path.insert(0, str(V2))
    parent_module = importlib.import_module('nano_lower_floor_rtx_runner_v2')
    parent, _ = parent_module.prelaunch_check(branch['source_run_id'], require_launch=False)
    if require_launch and (not branch['launch_allowed'] or
                           os.environ.get('LORACAST_REPEAT_CONTENTION_RTX_APPROVED') != '1'):
        raise RuntimeError('branch RTX launch gate is closed')
    return parent, {'policy': 'LoRACast-Exposure', 'run_id': run_id}


def import_native_and_core(manifest: dict):
    sys.path.insert(0, str(CORE_DIR))
    sys.path.insert(0, str(SOURCE))
    sys.path.insert(0, str(manifest['controller']['original_source_root']))
    core = importlib.import_module('run_qwen_training')
    native_p = importlib.import_module('nano_policy_variants').NanoPyramidBounded
    native_c = importlib.import_module('nano_final_exposure').NanoFinalExposure
    resume = importlib.import_module('policy_resume')
    from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, RecordDict
    return core, native_p, native_c, resume, ArrayRecord, ConfigRecord, MetricRecord, RecordDict


def make_native_strategy(manifest: dict, run: dict, native_p, native_c, out: Path):
    log_dir = out / 'native_flower_log'
    log_dir.mkdir(parents=True, exist_ok=True)
    common = dict(fraction_train=1.0, fraction_evaluate=1.0,
                  min_available_nodes=2, log_dir=str(log_dir), ROOT=str(out),
                  model_name=manifest['run_common']['model'], latte=False,
                  model_info={}, device_lookup={1: NANO, 2: NX},
                  device_profiles_path=None, tokens_per_example=256.0,
                  enable_deadline_guard=False, participation_guard_mode='per-client',
                  participation_fairness_rho=0.6,
                  aggregation_mode='processed_samples', experiment_id=run['run_id'],
                  rebalancing_enabled=False, expected_manifest_sha256=EXPECTED_STRONG_SHA,
                  expected_manifest_client_count=2,
                  residual_calibration_enabled=True)
    if run['policy'] == 'PyramidFL-style':
        return native_p(base_quota=320, min_quota=320, max_scale=600/320,
                        speed_window=5, scale_epochs_after_quota=False,
                        enable_rpi_only_rounds=False, use_token_ratio=False,
                        **common)
    cfg = manifest['controller']['native_train_config'][run['policy']]
    nano_keys = ('reference-nano-samples', 'reference-nano-steps',
                 'reference-nano-correction-max-samples',
                 'reference-nano-correction-max-steps',
                 'reference-nano-correction-max-seconds')
    return native_c(exposure_max=2.0, nx_eligible_unique=18441,
                    nx_minimum_samples=0, nx_maximum_samples=600,
                    nx_maximum_epochs=1, nano_config={k: cfg[k] for k in nano_keys},
                    comm_ema_alpha=.3, comm_safety_factor=1.0,
                    use_comm_prediction=False, static_allocation_after_round=0,
                    enable_rpi_only_rounds=True, use_token_ratio=True, **common)


def native_train_config(manifest: dict, run: dict, ConfigRecord):
    payload = dict(manifest['controller']['native_train_config'][run['policy']])
    payload['experiment-id'] = run['run_id']
    payload['aggregation-mode'] = 'processed_samples'
    payload['partition-manifest-path'] = str(SOURCE / 'loracast_strong.json')
    payload['sample-selection-mode'] = 'exposure_aware'
    return ConfigRecord({k: v for k, v in payload.items()
                         if isinstance(v, (str, int, float, bool))})


def actual_workload(round_config, name: str, policy: str) -> tuple[int, int, int]:
    device = NANO if name == 'nano' else NX
    s = int(round_config.get(f'max_samples_per_round_{device}', -1))
    e = int(round_config.get(f'local_epochs_{device}', -1))
    h = int(round_config.get(f'local_steps_{device}', -1))
    if h < 0 or not ((s == 0 and e == 0 and h == 0) or
                     (1 <= s <= 600 and e == 1)):
        raise ValueError(f'invalid native S/H/E for {policy}/{name}: {(s, e, h)}')
    if name == 'nano' and (s, e, h) not in ((320, 1, 0), (320, 1, 320)):
        raise ValueError('native Nano reference workload changed')
    # The exposure controller may choose a positive partial quota below 256
    # when less than one full candidate remains in the cumulative budget.
    if name == 'nx' and s and s < 256 and round_config.get('round_mode') != 'BOTH':
        raise ValueError('unexpected NX partial quota mode')
    if h > s * e:
        raise ValueError('native step cap exceeds selected sample/epoch work')
    return s, e, h


def pretraining_decision_gate(strategy, round_config, decision: dict,
                              current_nx_samples: int, round_number: int) -> dict:
    """Check the final Flower request before either client trains.

    The original allocator's search can return its base S with H=0 and a
    *reported* predicted time of zero when the first candidate is infeasible.
    Recompute both costs from the actual final ConfigRecord and the native
    predictor.  Server-side search/fallback runtime is outside every edge
    throughput and energy metric; only the returned device work matters.
    """
    if int(strategy.jetson_base_quota) != 256:
        raise ValueError(f'R{round_number} native NX search floor changed')
    nano_s, nano_e, nano_h = decision['nano']
    nx_s, nx_e, nx_h = decision['nx']
    if (nano_s, nano_e, nano_h) not in ((320, 1, 0), (320, 1, 320)):
        raise ValueError(f'R{round_number} Nano reference changed')
    if nx_s and nx_s > 600 or nx_e not in (0, 1):
        raise ValueError(f'R{round_number} invalid final NX quota or epoch')
    executed_nx = min(nx_s * nx_e, nx_h if nx_h > 0 else nx_s * nx_e)
    remaining = 2 * 18441 - int(current_nx_samples)
    if remaining < 0 or executed_nx > remaining:
        raise ValueError(f'R{round_number} NX exposure cap would be exceeded before training')
    native_decision = dict(strategy.last_nano_decision)
    fallback = native_decision.get('normal_fallback_reason')
    if fallback and fallback != 'base_candidate_outside_compute_window':
        raise ValueError(f'R{round_number} unrecognized native fallback reason: {fallback}')
    reference = float(strategy._predict_train_time(NANO, float(nano_s), round_config))
    predicted_nx = (float(strategy._predict_train_time(NX, float(nx_s), round_config))
                    if nx_s else 0.0)
    limit = reference * float(strategy.jetson_target_margin)
    if round_number > int(strategy.policy_warmup_rounds):
        if not (math.isfinite(reference) and math.isfinite(predicted_nx) and
                reference > 0 and (predicted_nx > 0 or nx_s == 0)):
            raise ValueError(f'R{round_number} final workload has no valid pre-round cost prediction')
        if nx_s and (fallback or predicted_nx > limit + 1e-9):
            raise ValueError(
                f'R{round_number} final NX workload exceeds predicted 0.95 Nano window: '
                f'S={nx_s} H={nx_h} NX={predicted_nx:.6f}s '
                f'reference={reference:.6f}s allowed={limit:.6f}s '
                f'fallback={fallback!r}')
    return {'nano_reference_s': reference, 'nx_final_s': predicted_nx,
            'allowed_s': limit, 'nx_remaining_exposure_samples_before': remaining,
            'nx_expected_executed_samples': executed_nx,
            'native_fallback_reason': fallback,
            'postwarmup_window_gate': ('PASS' if round_number > int(strategy.policy_warmup_rounds)
                                       else 'WARMUP_NOT_APPLICABLE')}


def make_reply(message, name: str, local_state: dict, metric: dict,
               edge: dict, ArrayRecord, MetricRecord, RecordDict):
    device = NANO if name == 'nano' else NX
    uid = 'nano-1' if name == 'nano' else 'nx-1'
    if metric['samples'] <= 0 or edge['support'] == 'unsupported':
        raise ValueError('a source reply requires completed and supported local work')
    m = {'device_name_chars': [ord(c) for c in device],
         'device_uid_chars': [ord(c) for c in uid],
         'manifest_sha256_chars': [ord(c) for c in EXPECTED_STRONG_SHA],
         'num-examples': int(metric['samples']),
         'dataset_size': 28143 if name == 'nano' else 18441,
         'partition_id': 1 if name == 'nano' else 0,
         'eligible_unique_samples': 28143 if name == 'nano' else 18441,
         'visited_sample_ids': [int(i) for i in metric['visited_sample_ids']],
         'local_steps': int(metric['optimizer_steps']),
         'input_tokens': int(metric['input_tokens']),
         'supervised_tokens': int(metric['supervised_tokens']),
         'train_loss': float(metric['train_loss']),
         'train_time': float(edge['compute_s']),
         'train_init_time': 0.0,
         'train_loop_time': float(edge['compute_s']),
         'train_energy': float(edge['compute_energy_j']),
         'client_upload_elapsed_time': 0.0,
         'upload_start_time': 0.0,
         'lora_rank': 16}
    return message.create_reply(RecordDict({
        'arrays': ArrayRecord(torch_state_dict=local_state),
        'metrics': MetricRecord(m)}))


def check_native_aggregate(native_state: dict, expected: dict) -> None:
    if set(native_state) != set(expected):
        raise ValueError('Flower aggregate adapter keys differ from RTX reference')
    import torch
    for key in expected:
        a, b = native_state[key].detach().cpu().float(), expected[key].detach().cpu().float()
        if a.shape != b.shape or not torch.allclose(a, b, rtol=1e-5, atol=1e-6):
            raise ValueError(f'Flower processed-sample aggregate differs for {key}')


def source_preflight(branch: dict) -> tuple[Path, dict, list, list, dict, dict]:
    """Verify the completed parent and reconstruct exactly the R49 sampler state."""
    source = Path(branch['source_directory'])
    if source.name != branch['source_run_id'] or source.is_symlink():
        raise ValueError('source directory identity changed')
    for relative, expected in branch['source_files_sha256'].items():
        path = source / relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError(f'frozen parent file missing or changed: {path}')
    source_state = json.loads((source / 'state.json').read_text())
    if (source_state['round'] != 100 or source_state['run_id'] != branch['source_run_id'] or
            source_state['protocol_sha256'] != branch['source_manifest_sha256']):
        raise ValueError('parent run identity or R100 state changed')
    rows = load_jsonl(source / 'rounds.jsonl')
    evaluations = load_jsonl(source / 'validation.jsonl')
    if [x['round'] for x in rows] != list(range(1, 101)) or \
       [x['round'] for x in evaluations] != list(range(0, 101, 10)):
        raise ValueError('parent history or evaluations are incomplete')
    if (rows[48]['checkpoint_sha256'] != branch['source_files_sha256']['adapter_49.pt'] or
            rows[49]['global_start_checkpoint_sha256'] != rows[48]['checkpoint_sha256']):
        raise ValueError('R49–R50 parent checkpoint chain changed')
    metadata = json.loads((source / 'native_flower_log/resume_state/round_49.json').read_text())
    if (metadata['completed_round'] != 49 or metadata['experiment_id'] != branch['source_run_id'] or
            metadata['checkpoint'] != str(source / 'adapter_49.pt') or
            metadata['checkpoint_sha256'] != rows[48]['checkpoint_sha256'] or
            metadata['state_sha256'] != rows[48]['native_strategy_state_sha256'] or
            metadata['state_sha256'] != branch['source_files_sha256']['native_flower_log/resume_state/round_49.pkl']):
        raise ValueError('R49 native Flower snapshot identity changed')
    replay = {'nano': Counter(), 'nx': Counter()}
    r49 = None
    previous = evaluations[0]['checkpoint_sha256']
    for row in rows:
        if row['global_start_checkpoint_sha256'] != previous:
            raise ValueError(f'parent checkpoint chain breaks at R{row["round"]}')
        previous = row['checkpoint_sha256']
        for name in ('nano', 'nx'):
            metric = row['clients'][name]
            visited = metric['visited_sample_ids']
            if metric['samples'] != len(visited) or metric['optimizer_steps'] != metric['samples']:
                raise ValueError(f'parent executed work differs from sample visits at R{row["round"]}/{name}')
            replay[name].update(str(sample) for sample in visited)
        if row['round'] == 49:
            r49 = {name: dict(replay[name]) for name in ('nano', 'nx')}
    if previous != source_state['checkpoint_sha256'] or \
       source_state['native_strategy_state_sha256'] != rows[-1]['native_strategy_state_sha256']:
        raise ValueError('parent R100 state differs from history')
    if dict(replay['nano']) != source_state['nano_visits'] or \
       dict(replay['nx']) != source_state['visits']:
        raise ValueError('replayed visits differ from parent R100 state')
    if abs(rows[48]['nx_cumulative_exposure'] - sum(r49['nx'].values()) / 18441) > 1e-12:
        raise ValueError('R49 NX exposure differs from replayed visits')
    return source, source_state, rows, evaluations, r49['nx'], r49['nano']


def copy_csv_prefix(original: Path, target: Path) -> None:
    with original.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f'parent native CSV has no header: {original}')
        fields = reader.fieldnames
        rows = [row for row in reader if int(row['round']) <= 49]
    if not rows or max(int(row['round']) for row in rows) != 49:
        raise ValueError(f'parent native CSV prefix does not reach R49: {original}')
    with target.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def construct_in_scratch(manifest, run, native_p, native_c, out):
    """Native constructor writes CSVs; keep it away from committed branch logs."""
    scratch = out / 'constructor_scratch'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='native_', dir=scratch) as folder:
        strategy = make_native_strategy(manifest, run, native_p, native_c, Path(folder))
    return strategy


def relocate_native(strategy, source: Path, out: Path, new_run_id: str) -> None:
    expected = {'log_dir', 'aggregation_path', 'csv_path'}
    source_paths = {key: value for key, value in vars(strategy).items()
                    if isinstance(value, (str, os.PathLike)) and
                    str(value).startswith(str(source) + os.sep)}
    if set(source_paths) != expected:
        raise ValueError(f'unknown or missing source native output paths: {source_paths}')
    native_dir = out / 'native_flower_log'
    native_dir.mkdir(parents=True, exist_ok=True)
    target_csv = native_dir / Path(strategy.csv_path).name
    target_aggregation = native_dir / 'aggregation.csv'
    if not target_csv.exists():
        copy_csv_prefix(Path(strategy.csv_path), target_csv)
    if not target_aggregation.exists():
        copy_csv_prefix(Path(strategy.aggregation_path), target_aggregation)
    strategy.log_dir = str(native_dir)
    strategy.csv_path = str(target_csv)
    strategy.aggregation_path = str(target_aggregation)
    strategy.experiment_id = new_run_id
    for key, value in vars(strategy).items():
        if isinstance(value, (str, os.PathLike)) and str(value).startswith(str(source) + os.sep):
            raise ValueError(f'native state still points to parent: {key}')
    for key in expected:
        if not Path(getattr(strategy, key)).resolve().is_relative_to(out.resolve()):
            raise ValueError(f'native {key} points outside branch output')


def restore_parent_r49(manifest, run, source, out, native_p, native_c, resume):
    parent_run = {'policy': run['policy'], 'run_id': branch_manifest()['source_run_id']}
    strategy = construct_in_scratch(manifest, parent_run, native_p, native_c, out)
    strategy.log_dir = str(source / 'native_flower_log')
    resume.load_completed_round(strategy, 49, str(source / 'adapter_49.pt'))
    if sorted(strategy.round_stats) != list(range(1, 50)) or \
       sorted(strategy.round_configs) != list(range(1, 50)):
        raise ValueError('parent native Flower state lacks R1–R49')
    relocate_native(strategy, source, out, run['run_id'])
    return strategy


def verify_branch(out: Path, branch: dict, state: dict, history: list, evaluations: list, core) -> None:
    if (state['run_id'] != branch['branch_run_id'] or
            state['protocol_sha256'] != digest(MANIFEST) or
            not 49 <= state['round'] <= 100):
        raise ValueError('branch resume identity changed')
    if digest(out / f'adapter_{state["round"]}.pt') != state['checkpoint_sha256']:
        raise ValueError('branch current checkpoint hash changed')
    snapshot = out / 'native_flower_log' / 'resume_state' / f'round_{state["round"]:02d}.pkl'
    if digest(snapshot) != state['native_strategy_state_sha256']:
        raise ValueError('branch native Flower state hash changed')
    if len(history) != state['round'] or len(evaluations) != 1 + state['round'] // 10:
        raise ValueError('branch round or evaluation history length changed')
    core.verify_committed_run(out, state, history, evaluations)


def reject_uncommitted_artifacts(out: Path, current: int) -> None:
    """Fail closed after an interrupted round; preserve evidence for audit."""
    later = current + 1
    suspect = [out / f'adapter_{later}.pt',
               out / f'adapter_{later}.pt.tmp',
               out / f'native_flower_round_{later:03d}.log',
               out / f'unsupported_feedback_round_{later:03d}.json',
               out / 'native_flower_log/resume_state' / f'round_{later:02d}.pkl',
               out / 'native_flower_log/resume_state' / f'round_{later:02d}.json']
    if any(path.exists() for path in suspect):
        raise ValueError(f'uncommitted R{later} artifact exists; manual audit required')
    native = out / 'native_flower_log'
    for path in native.glob('*.csv'):
        with path.open(newline='') as stream:
            if any(int(row['round']) > current for row in csv.DictReader(stream)):
                raise ValueError(f'uncommitted native CSV row exists: {path}')
    for path in native.glob('*round_*.json'):
        try:
            rnd = int(path.stem.rsplit('_', 1)[-1])
        except ValueError:
            continue
        if rnd > current:
            raise ValueError(f'uncommitted native decision exists: {path}')


def bootstrap_branch(manifest, run, branch, source, rows, evaluations,
                     nx_visits, nano_visits, out, native_p, native_c, resume, core):
    if out.exists():
        raise FileExistsError(f'refusing to overwrite existing branch directory: {out}')
    out.mkdir(parents=True)
    for rnd in (0, 10, 20, 30, 40, 49):
        name = f'adapter_{rnd}.pt'
        shutil.copy2(source / name, out / name)
        if digest(out / name) != branch['source_files_sha256'][name]:
            raise ValueError(f'bootstrap checkpoint copy differs: {name}')
    prefix_rows = rows[:49]
    prefix_evaluations = [row for row in evaluations if row['round'] <= 49]
    (out / 'rounds.jsonl').write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in prefix_rows))
    (out / 'validation.jsonl').write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in prefix_evaluations))
    strategy = restore_parent_r49(manifest, run, source, out, native_p, native_c, resume)
    if (strategy.nx_cumulative_samples != sum(nx_visits.values()) or
            strategy.nx_seen_ids != {int(k) for k in nx_visits} or
            strategy.nx_unique_visited != len(nx_visits)):
        raise ValueError('R49 native exposure differs from replayed sample visits')
    resume.save_completed_round(strategy, 49, str(out / 'adapter_49.pt'))
    snapshot = out / 'native_flower_log/resume_state/round_49.pkl'
    state = {'run_id': run['run_id'], 'round': 49,
             'checkpoint_sha256': digest(out / 'adapter_49.pt'),
             'native_strategy_state_sha256': digest(snapshot),
             'protocol_sha256': digest(MANIFEST),
             'source_r49_checkpoint_sha256': branch['source_files_sha256']['adapter_49.pt'],
             'source_r49_native_state_sha256': branch['source_files_sha256']['native_flower_log/resume_state/round_49.pkl'],
             'visits': nx_visits, 'nano_visits': nano_visits}
    core.atomic_json(out / 'state.json', state)
    verify_branch(out, branch, state, prefix_rows, prefix_evaluations, core)
    restored = construct_in_scratch(manifest, run, native_p, native_c, out)
    restored.log_dir = str(out / 'native_flower_log')
    resume.load_completed_round(restored, 49, str(out / 'adapter_49.pt'))
    if restored.nx_cumulative_samples != strategy.nx_cumulative_samples or \
       restored.experiment_id != run['run_id']:
        raise ValueError('branched R49 native resume parity failed')
    return state, strategy, prefix_rows, prefix_evaluations


def cpu_r50_preflight(manifest, run, branch, source, rows, nx_visits, nano_visits,
                      core, native_p, native_c, resume, ArrayRecord, ConfigRecord):
    """Replay the first normal decision and sample selection without learning."""
    import torch
    strong = json.loads((SOURCE / 'loracast_strong.json').read_text())
    partitions = json.loads(PARTITIONS.read_text())
    populations = {'nano': partitions['pi_stationary'],
                   'nx': partitions['nx_budgets']['DFull']}
    maps = {'nano': {int(sample): i for i, sample in enumerate(strong['clients']['1']['train_indices'])},
            'nx': {int(sample): i for i, sample in enumerate(strong['clients']['0']['train_indices'])}}
    source_r50 = rows[49]
    parent_checks = [source / 'state.json', source / 'adapter_49.pt',
                     source / 'native_flower_log/aggregation.csv']
    before = {str(path): digest(path) for path in parent_checks}
    with tempfile.TemporaryDirectory(prefix='r49_bootstrap_cpu_', dir=HERE) as folder:
        bootstrap_out = Path(folder) / branch['branch_run_id']
        state, boot_strategy, history, evaluations = bootstrap_branch(
            manifest, run, branch, source, rows,
            load_jsonl(source / 'validation.jsonl'), nx_visits, nano_visits,
            bootstrap_out, native_p, native_c, resume, core)
        if state['round'] != 49 or len(history) != 49 or len(evaluations) != 5 or \
           boot_strategy.experiment_id != run['run_id']:
            raise ValueError('isolated CPU bootstrap did not commit R49 correctly')
    with tempfile.TemporaryDirectory(prefix='r50_cpu_', dir=HERE) as folder:
        out = Path(folder)
        strategy = restore_parent_r49(manifest, run, source, out, native_p, native_c, resume)
        if strategy.nx_cumulative_samples != sum(nx_visits.values()) or \
           strategy.nx_seen_ids != {int(k) for k in nx_visits}:
            raise ValueError('R49 Flower exposure does not match sampler replay')
        global_state = torch.load(source / 'adapter_49.pt', map_location='cpu', weights_only=True)
        config = native_train_config(manifest, run, ConfigRecord)
        messages = list(strategy.configure_train(
            50, ArrayRecord(torch_state_dict=global_state), config, TwoClientGrid()))
        round_config = strategy.round_configs[50]
        names = {'nano' if m.metadata.dst_node_id == 1 else 'nx' for m in messages}
        if names != set(source_r50['source_flower_destinations']):
            raise ValueError('R50 Flower participants differ from parent normal round')
        decision = {name: actual_workload(round_config, name, run['policy'])
                    for name in ('nano', 'nx')}
        if str(round_config.get('round_mode')) != source_r50['source_round_mode']:
            raise ValueError('R50 native round mode differs from parent normal round')
        pretraining_decision_gate(strategy, round_config, decision,
                                  sum(nx_visits.values()), 50)
        for name, partition, visits in (('nano', 1, nano_visits), ('nx', 0, nx_visits)):
            expected = source_r50['requested_workload'][name]
            if decision[name] != (expected['S'], expected['E'], expected['H']):
                raise ValueError(f'R50 native allocation parity failed for {name}')
            if name in names:
                seed = 42 + 1000003 * 50 + partition
                _, selected = core.choose_positions(populations[name], maps[name],
                                                     decision[name][0], visits, seed)
            else:
                selected = []
            if selected != source_r50['clients'][name]['selected_sample_ids']:
                raise ValueError(f'R50 sample-ID parity failed for {name}')
    if any(digest(Path(raw)) != expected for raw, expected in before.items()):
        raise ValueError('parent file changed during CPU preflight')
    return {'status': 'R49_BOOTSTRAP_AND_R50_DECISION_SAMPLE_PARITY_PASS',
            'source_round': 49, 'first_branch_round': 50,
            'r50_requested_workload': {name: dict(zip(('S', 'E', 'H'), decision[name]))
                                       for name in ('nano', 'nx')},
            'r49_nx_executed_samples': sum(nx_visits.values()),
            'gpu_work_performed': False}


def execute(run_id: str, stop_after_round: int | None = None):
    manifest, run = prelaunch_check(run_id, require_launch=True)
    branch = branch_manifest()
    source, _, source_rows, source_evaluations, source_nx_visits, source_nano_visits = source_preflight(branch)
    core, native_p, native_c, resume, ArrayRecord, ConfigRecord, MetricRecord, RecordDict = import_native_and_core(manifest)
    cpu_r50_preflight(manifest, run, branch, source, source_rows,
                      source_nx_visits, source_nano_visits,
                      core, native_p, native_c, resume, ArrayRecord, ConfigRecord)
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            check=True, capture_output=True, text=True)
    if active.stdout.strip():
        raise RuntimeError(f'GPU has an active compute process: {active.stdout.strip()}')
    from datasets import load_dataset
    from e2fl.runtime.llm_data import load_data
    from e2fl.runtime.partition_manifest import dataset_digest, load_manifest
    import torch

    torch.set_num_threads(4)
    strong = load_manifest(SOURCE / 'loracast_strong.json')
    data = load_dataset(strong['dataset_name'], split='train', revision=strong['dataset_revision'])
    if len(data) != strong['dataset_size'] or dataset_digest(data) != strong['dataset_sha256']:
        raise ValueError('Strong dataset identity mismatch')
    partitions = json.loads(PARTITIONS.read_text())
    nano_population = partitions['pi_stationary']
    nx_population = partitions['nx_budgets']['DFull']
    if len(nano_population) != 28143 or len(nx_population) != 18441:
        raise ValueError('eligible Strong client counts changed')
    core.PI_GLOBAL_IDS = strong['clients']['1']['train_indices']
    core.NX_GLOBAL_IDS = strong['clients']['0']['train_indices']
    core.PI_GLOBAL_MAP = {idx: pos for pos, idx in enumerate(core.PI_GLOBAL_IDS)}
    core.NX_GLOBAL_MAP = {idx: pos for pos, idx in enumerate(core.NX_GLOBAL_IDS)}
    bases = {}
    for name, partition in (('pi', 1), ('nx', 0)):
        bases[name], _ = load_data('yahma/alpaca-cleaned', partition, 2, 1,
                                   'hf:' + core.MODEL, 64,
                                   str(SOURCE / 'loracast_strong.json'))
    model, evaluator = core.create_models()
    views, batches = core.make_eval(evaluator, data, strong)

    OUT_ROOT.mkdir(exist_ok=True)
    out = OUT_ROOT / run_id
    state_path = out / 'state.json'
    rounds_path = out / 'rounds.jsonl'
    eval_path = out / 'validation.jsonl'
    protocol_sha = digest(MANIFEST)
    config = native_train_config(manifest, run, ConfigRecord)
    factor_schedule = {int(row['round']): row for row in branch['factor_rounds']}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['run_id'] != run_id or state['protocol_sha256'] != protocol_sha:
            raise ValueError('resume protocol identity mismatch')
        current = int(state['round'])
        history = load_jsonl(rounds_path)
        evaluations = load_jsonl(eval_path)
        if len(history) != current:
            raise ValueError('uncommitted round history present; manual integrity audit required')
        verify_branch(out, branch, state, history, evaluations, core)
        reject_uncommitted_artifacts(out, current)
        checkpoint = out / f'adapter_{current}.pt'
        global_state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        strategy = construct_in_scratch(manifest, run, native_p, native_c, out)
        strategy.log_dir = str(out / 'native_flower_log')
        resume.load_completed_round(strategy, current, str(checkpoint))
        for name in ('log_dir', 'aggregation_path', 'csv_path'):
            if not Path(getattr(strategy, name)).resolve().is_relative_to(out.resolve()):
                raise ValueError(f'restored native {name} points outside branch')
        visits, nano_visits = state['visits'], state['nano_visits']
    else:
        state, strategy, history, evaluations = bootstrap_branch(
            manifest, run, branch, source, source_rows, source_evaluations,
            source_nx_visits, source_nano_visits, out, native_p, native_c, resume, core)
        current = 49
        visits, nano_visits = state['visits'], state['nano_visits']
        global_state = torch.load(out / 'adapter_49.pt', map_location='cpu', weights_only=True)
    if current == 100:
        print(json.dumps({'run_id': run_id, 'status': 'already_complete_and_verified'}))
        return
    max_round = stop_after_round if stop_after_round is not None else 100
    if current > max_round:
        raise ValueError('committed round exceeds smoke stop')
    for rnd in range(current + 1, max_round + 1):
        shared_start_sha = digest(out / f'adapter_{rnd-1}.pt')
        stdout_path = out / f'native_flower_round_{rnd:03d}.log'
        with stdout_path.open('w') as stream, contextlib.redirect_stdout(stream):
            messages = list(strategy.configure_train(
                rnd, ArrayRecord(torch_state_dict=global_state), config, TwoClientGrid()))
        round_config = strategy.round_configs[rnd]
        expected_names = {'nano', 'nx'} if run['policy'] == 'PyramidFL-style' else (
            {'nano'} if round_config.get('round_mode') == 'PROTECTED_ONLY' else {'nano', 'nx'})
        message_names = {'nano' if m.metadata.dst_node_id == 1 else 'nx'
                         for m in messages if m.metadata.dst_node_id in (1, 2)}
        if len(messages) != len(message_names) or message_names != expected_names:
            raise ValueError(f'Flower destinations disagree with native action at R{rnd}')
        decision = {name: actual_workload(round_config, name, run['policy'])
                    for name in ('nano', 'nx')}
        if (decision['nx'][0] == 0) != ('nx' not in message_names):
            raise ValueError('omission message and NX quota disagree')
        pre_round_prediction = pretraining_decision_gate(
            strategy, round_config, decision, sum(int(v) for v in visits.values()), rnd)
        phase = factor_schedule[rnd]['phase'] if rnd in factor_schedule else 'normal'
        nano_factor = (float(factor_schedule[rnd]['nano_compute_seconds_per_token_factor'])
                       if rnd in factor_schedule else 1.0)

        states, metrics, projections, replies = {}, {}, {}, []
        for name, client_key, population, visit_counter, node_id in (
            ('nano', 'pi', nano_population, nano_visits, 1),
            ('nx', 'nx', nx_population, visits, 2)):
            quota, epochs, step_cap = decision[name]
            if name not in message_names:
                metrics[name] = {'samples': 0, 'optimizer_steps': 0,
                                 'input_tokens': 0, 'supervised_tokens': 0,
                                 'train_loss': None, 'rtx_train_s': 0.0,
                                 'selected_sample_ids': [], 'visited_sample_ids': [],
                                 'completed_passes': 0.0,
                                 'local_training_start_checkpoint_sha256': None}
                projections[name] = project(name, 0)
                continue
            local, metric = core.train_client(
                model, global_state, bases[client_key],
                core.PI_GLOBAL_MAP if name == 'nano' else core.NX_GLOBAL_MAP,
                population, visit_counter, client_key, rnd, quota, epochs, step_cap)
            if local is None:
                raise ValueError('Flower selected a client but RTX trainer returned no update')
            expected_steps = min(quota * epochs,
                                 step_cap if step_cap > 0 else quota * epochs)
            if metric['samples'] != expected_steps or metric['optimizer_steps'] != expected_steps:
                raise ValueError('executed S/H/E differs from native request')
            metric['local_training_start_checkpoint_sha256'] = shared_start_sha
            states[name], metrics[name] = local, metric
            try:
                normal_projection = project(name, int(metric['input_tokens']))
            except ValueError as exc:
                core.atomic_json(out / f'unsupported_feedback_round_{rnd:03d}.json',
                    {'round': rnd, 'device': name, 'actual_input_tokens': metric['input_tokens'],
                     'reason': str(exc), 'last_committed_round': current})
                raise
            feedback = dict(normal_projection)
            if name == 'nano':
                feedback['compute_s'] = float(normal_projection['compute_s']) * nano_factor
            # Only Nano compute time is changed for next-round native feedback. The
            # normal-profile energy remains in the native reply solely to
            # satisfy the existing Flower metric schema; it is unsupported as
            # contended energy and is never reported as an energy result.
            projections[name] = {
                **feedback,
                'compute_energy_j': (None if name == 'nano' and phase != 'normal'
                                     else float(normal_projection['compute_energy_j'])),
                'normal_profile_compute_s': float(normal_projection['compute_s']),
                'normal_profile_compute_energy_j': float(normal_projection['compute_energy_j']),
                'nano_time_factor': nano_factor if name == 'nano' else 1.0,
                'phase': phase,
                'cost_boundary': ('trace_projected_local_compute_time_only'
                                  if phase != 'normal' else
                                  'trace_projected_normal_local_compute_time_and_energy'),
                'contention_energy_status': ('unsupported' if phase != 'normal'
                                             else 'normal_profile_supported')}
            message = next(m for m in messages if m.metadata.dst_node_id == node_id)
            replies.append(make_reply(message, name, local, metric, feedback,
                                      ArrayRecord, MetricRecord, RecordDict))
        if len(replies) != len(messages):
            raise ValueError('one native reply is required per trained client')
        with stdout_path.open('a') as stream, contextlib.redirect_stdout(stream):
            native_arrays, _ = strategy.aggregate_train(rnd, replies)
        if native_arrays is None or sorted(strategy.round_stats) != list(range(1, rnd + 1)):
            raise ValueError('original Flower aggregation/state commit failed')
        total_samples = sum(metric['samples'] for metric in metrics.values())
        weights = {name: metric['samples'] / total_samples for name, metric in metrics.items()}
        expected_aggregate = core.aggregate(states, {name: weights[name] for name in states})
        updated = native_arrays.to_torch_state_dict()
        check_native_aggregate(updated, expected_aggregate)
        if run['policy'] == 'LoRACast-Exposure':
            cumulative = sum(int(row['clients']['nx']['samples']) for row in history) + metrics['nx']['samples']
            if strategy.nx_cumulative_samples != cumulative or cumulative > 2 * len(nx_population):
                raise ValueError('native cumulative exposure state differs from actual executed work')
            unique = set(int(k) for k, v in visits.items() if v > 0)
            if strategy.nx_seen_ids != unique or strategy.nx_unique_visited != len(unique):
                raise ValueError('native sample-ID exposure state differs from actual visits')
        checkpoint = out / f'adapter_{rnd}.pt'
        core.atomic_state(checkpoint, updated)
        resume.save_completed_round(strategy, rnd, str(checkpoint))
        snapshot = Path(strategy.log_dir) / 'resume_state' / f'round_{rnd:02d}.pkl'
        record = {'run_id': run_id, 'round': rnd, 'policy': run['policy'],
            'policy_variant': branch['policy_variant'],
            'phase': phase,
            'fig9_reference_round': (factor_schedule[rnd]['fig9_reference_round']
                                     if rnd in factor_schedule else None),
            'nano_compute_seconds_per_token_factor': nano_factor,
            'pre_round_prediction': pre_round_prediction,
            'global_start_checkpoint_sha256': shared_start_sha,
            'checkpoint_sha256': digest(checkpoint),
            'requested_workload': {name: {'S': w[0], 'E': w[1], 'H': w[2]}
                                   for name, w in decision.items()},
            'source_round_mode': str(round_config.get('round_mode')),
            'source_flower_destinations': sorted(message_names),
            'clients': metrics, 'aggregation_weights': weights,
            'nx_cumulative_exposure': sum(visits.values()) / len(nx_population),
            'edge_compute_projection': projections,
            'native_strategy_state_sha256': digest(snapshot),
            'measurement': ('RTX_actual_training_and_validation; native_Flower_decision_and_aggregation; '
                            'trace_projected_edge_local_compute_time; Nano_repeated_R50_R100_factor; '
                            'stress_energy_unsupported; server_decision_overhead_excluded')}
        append_jsonl(rounds_path, record)
        if rnd in CHECKPOINT_ROUNDS:
            evaluated = core.evaluate(evaluator, updated, views, batches,
                                      run_id, rnd, checkpoint)
            evaluated['nano_associated_loss'] = evaluated['pi_loss']
            evaluated['nano_partition_origin'] = 'original Strong Pi5 client 1'
            append_jsonl(eval_path, evaluated)
        core.atomic_json(state_path, {'run_id': run_id, 'round': rnd,
            'checkpoint_sha256': digest(checkpoint),
            'native_strategy_state_sha256': digest(snapshot),
            'protocol_sha256': protocol_sha,
            'source_r49_checkpoint_sha256': branch['source_files_sha256']['adapter_49.pt'],
            'source_r49_native_state_sha256': branch['source_files_sha256']['native_flower_log/resume_state/round_49.pkl'],
            'visits': visits, 'nano_visits': nano_visits})
        global_state = updated
        history.append(record)
        if stop_after_round == 50 and rnd == 50:
            verify_branch(out, branch, json.loads(state_path.read_text()),
                          history, load_jsonl(eval_path), core)
            restored = construct_in_scratch(manifest, run, native_p, native_c, out)
            restored.log_dir = str(out / 'native_flower_log')
            resume.load_completed_round(restored, 50, str(checkpoint))
            if restored.round_stats != strategy.round_stats or (
                    run['policy'] == 'LoRACast-Exposure' and
                    restored.nx_cumulative_samples != strategy.nx_cumulative_samples):
                raise ValueError('R50 native strategy resume parity failed')
            print(json.dumps({'run_id': run_id,
                'status': 'R50_NATIVE_FLOWER_SMOKE_AND_RESUME_PASS',
                'R49_sha256': digest(out / 'adapter_49.pt'),
                'R50_sha256': digest(checkpoint),
                'same_global_start_for_both_clients': all(
                    m['local_training_start_checkpoint_sha256'] == shared_start_sha
                    for m in metrics.values() if m['samples']),
                'native_aggregation_weights': weights,
                'native_state_snapshot_sha256': digest(snapshot)}), flush=True)
        if rnd % 10 == 0:
            print(json.dumps({'run_id': run_id, 'round': rnd,
                'nx_exposure': record['nx_cumulative_exposure'],
                'native_round_stats': len(strategy.round_stats)}), flush=True)
    verify_branch(out, branch, json.loads(state_path.read_text()),
                  history, load_jsonl(eval_path), core)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--stop-after-round', type=int, choices=(50,), default=None)
    args = parser.parse_args()
    if args.validate_only:
        manifest, run = prelaunch_check(args.run_id, require_launch=False)
        branch = branch_manifest()
        source, _, rows, _, nx_visits, nano_visits = source_preflight(branch)
        core, native_p, native_c, resume, ArrayRecord, ConfigRecord, _, _ = import_native_and_core(manifest)
        result = cpu_r50_preflight(manifest, run, branch, source, rows,
                                   nx_visits, nano_visits, core, native_p, native_c,
                                   resume, ArrayRecord, ConfigRecord)
        result.update({'run_id': run['run_id'], 'policy': run['policy'],
                       'launch_allowed': branch['launch_allowed'],
                       'branch_manifest_sha256': digest(MANIFEST)})
        print(json.dumps(result, indent=2))
    else:
        with (HERE / 'branch_gpu.lock').open('a+') as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('another branch RTX process holds the GPU lock') from exc
            execute(args.run_id, args.stop_after_round)


if __name__ == '__main__':
    main()
