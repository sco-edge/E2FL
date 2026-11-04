#!/usr/bin/env python3
"""Isolated matched R0-R100 Nano+NX Strong RTX FL with native Flower strategy.

The strategy receives only this policy's previous completed work and the
predeclared normal-condition edge *local-compute* projection. The Flower
configure/aggregate hooks make the decisions and update native state. This is
an RTX learning experiment with simulated edge feedback, not an edge run.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The source protocol is read-only. On gpu_server, set LORACAST_NANO_PROTOCOL
# to its absolute directory before running this isolated copy.
_LOCAL_NANO_PROTOCOL = (HERE.parents[1] /
    'training_cost_projection' / 'edge_protocol')
NANO_PROTOCOL = Path(os.environ.get(
    'LORACAST_NANO_PROTOCOL', str(_LOCAL_NANO_PROTOCOL)))
sys.path.insert(0, str(NANO_PROTOCOL))
from edge_profile_oracle import FROZEN, digest, project
from nano_qwen_rtx_runner import (CHECKPOINT_ROUNDS, CORE_DIR, EXPECTED_INITIAL_SHA,
                                  EXPECTED_PARTITION_SHA, EXPECTED_STRONG_SHA,
                                  PARTITIONS, SOURCE, append_jsonl, load_jsonl)


MANIFEST = HERE / 'experiment_manifest.json'
HASHES = HERE / 'nano_adaptive_native_source_hashes_v1.json'
STRESS_SOURCE = HERE / 'source_repeated_contention_manifest_snapshot.json'
OUT_ROOT = HERE / 'runs'
NANO = 'jetson_orin_nano_super'
NX = 'jetson_orin_nx'


class TwoClientGrid:
    def get_node_ids(self):
        return [1, 2]


def prelaunch_check(run_id: str, *, require_launch: bool) -> tuple[dict, dict]:
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
    expected_factors = [
        {**{key: value for key, value in row.items() if key != 'round'},
         'extension_round': row['round']}
        for row in stress_source['factor_rounds']]
    if (digest(STRESS_SOURCE) != schedule['source_sha256'] or
            stress_source['schema_version'] != 1 or
            stress_source['first_branch_round'] != 50 or
            stress_source['last_round'] != 100 or
            factors != expected_factors or
            schedule['normal_rounds'] != [1, 49] or
            [int(row['extension_round']) for row in factors] != list(range(50, 101)) or
            any(not math.isfinite(float(row['nano_compute_seconds_per_token_factor'])) or
                float(row['nano_compute_seconds_per_token_factor']) <= 0 for row in factors)):
        raise ValueError('frozen matched Fig. 9 repeated schedule changed')
    for row in factors:
        offset = (int(row['extension_round']) - 50) % 12
        expected_phase = ('normal' if offset < 4 else
                          'contention' if offset < 8 else 'recovery')
        expected_reference = None if offset < 4 else (5 + (offset - 4)
                                                     if offset < 8 else 9 + (offset - 8))
        if row['phase'] != expected_phase or row['fig9_reference_round'] != expected_reference:
            raise ValueError('matched 4/4/4 phase or measured reference changed')
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
                           os.environ.get('LORACAST_MATCHED_RTX_APPROVED') != '1'):
        raise RuntimeError('matched RTX launch gate is closed')
    return manifest, matches[0]


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


def execute(run_id: str, stop_after_round: int | None = None):
    manifest, run = prelaunch_check(run_id, require_launch=True)
    core, native_p, native_c, resume, ArrayRecord, ConfigRecord, MetricRecord, RecordDict = import_native_and_core(manifest)
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

    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'state.json'
    rounds_path = out / 'rounds.jsonl'
    eval_path = out / 'validation.jsonl'
    if not state_path.exists() and any(out.iterdir()):
        raise ValueError('partial new-run artifacts exist without state.json; preserve and audit')
    protocol_sha = digest(MANIFEST)
    strategy = make_native_strategy(manifest, run, native_p, native_c, out)
    config = native_train_config(manifest, run, ConfigRecord)
    factor_schedule = {
        int(row['extension_round']): row for row in
        manifest['stress_schedule']['factor_rounds']}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['run_id'] != run_id or state['protocol_sha256'] != protocol_sha:
            raise ValueError('resume protocol identity mismatch')
        current = int(state['round'])
        history = load_jsonl(rounds_path)
        evaluations = load_jsonl(eval_path)
        if len(history) != current:
            raise ValueError('uncommitted round history present; manual integrity audit required')
        core.verify_committed_run(out, state, history, evaluations)
        checkpoint = out / f'adapter_{current}.pt'
        global_state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if current:
            resume.load_completed_round(strategy, current, str(checkpoint))
            snapshot = Path(strategy.log_dir) / 'resume_state' / f'round_{current:02d}.pkl'
            if digest(snapshot) != state['native_strategy_state_sha256']:
                raise ValueError('native strategy state hash mismatch')
        visits, nano_visits = state['visits'], state['nano_visits']
    else:
        current = 0
        history, visits, nano_visits = [], {}, {}
        global_state = torch.load(SOURCE / 'initial_adapter_round0.pt', map_location='cpu', weights_only=True)
        core.atomic_state(out / 'adapter_0.pt', global_state)
        initial = core.evaluate(evaluator, global_state, views, batches,
                                run_id, 0, out / 'adapter_0.pt')
        initial['nano_associated_loss'] = initial['pi_loss']
        initial['nano_partition_origin'] = 'original Strong Pi5 client 1'
        append_jsonl(eval_path, initial)
        core.atomic_json(state_path, {'run_id': run_id, 'round': 0,
            'checkpoint_sha256': digest(out / 'adapter_0.pt'),
            'protocol_sha256': protocol_sha, 'visits': visits, 'nano_visits': nano_visits})
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
            # Exactly as in the prior branch, only Nano compute time
            # is slowed for the native next-round estimator feedback.  The
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
            'policy_variant': 'LoRACast-Exposure-NX-floor-256-matched-repeat4x-full100',
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
                            'trace_projected_edge_local_compute_time; Nano_repeat4x_time_factor_R50_R100; '
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
            'visits': visits, 'nano_visits': nano_visits})
        global_state = updated
        history.append(record)
        if stop_after_round == 1 and rnd == 1:
            core.verify_committed_run(out, json.loads(state_path.read_text()),
                                      history, load_jsonl(eval_path))
            restored = make_native_strategy(manifest, run, native_p, native_c, out)
            resume.load_completed_round(restored, 1, str(checkpoint))
            if restored.round_stats != strategy.round_stats or (
                    run['policy'] == 'LoRACast-Exposure' and
                    restored.nx_cumulative_samples != strategy.nx_cumulative_samples):
                raise ValueError('R1 native strategy resume parity failed')
            print(json.dumps({'run_id': run_id,
                'status': 'R1_NATIVE_FLOWER_SMOKE_AND_RESUME_PASS',
                'R0_sha256': digest(out / 'adapter_0.pt'),
                'R1_sha256': digest(checkpoint),
                'same_global_start_for_both_clients': all(
                    m['local_training_start_checkpoint_sha256'] == shared_start_sha
                    for m in metrics.values() if m['samples']),
                'native_aggregation_weights': weights,
                'native_state_snapshot_sha256': digest(snapshot)}), flush=True)
        if rnd % 10 == 0:
            print(json.dumps({'run_id': run_id, 'round': rnd,
                'nx_exposure': record['nx_cumulative_exposure'],
                'native_round_stats': len(strategy.round_stats)}), flush=True)
    core.verify_committed_run(out, json.loads(state_path.read_text()),
                              history, load_jsonl(eval_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--stop-after-round', type=int, choices=(1,), default=None)
    args = parser.parse_args()
    if args.validate_only:
        manifest, run = prelaunch_check(args.run_id, require_launch=False)
        print(json.dumps({'status': 'ADAPTIVE_INPUT_SOURCE_HASH_PREFLIGHT_PASS',
            'run_id': run['run_id'], 'policy': run['policy'],
            'launch_allowed': manifest['adaptive_launch_allowed'],
            'gpu_work_performed': False}, indent=2))
    else:
        execute(args.run_id, args.stop_after_round)


if __name__ == '__main__':
    main()
