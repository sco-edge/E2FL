#!/usr/bin/env python3
"""Read-only integrity audit for the isolated LoRACast repeated-stress run."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_MANIFEST_SHA = 'c8a9b67c33ce900f58932d3a48267b3568499a7083c9b92780a8b5c4043b63de'
EXPECTED_RUNNER_SHA = '9402c9c5bb3bf48f678aaf22f859cbad79dec8224f24dfd17e4fbf87c69f321c'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def main() -> None:
    require(sha(HERE / 'experiment_manifest.json') == EXPECTED_MANIFEST_SHA,
            'frozen branch manifest changed')
    require(sha(HERE / 'branch_run_adaptive_training.py') == EXPECTED_RUNNER_SHA,
            'frozen branch run_adaptive_training changed')
    manifest = json.loads((HERE / 'experiment_manifest.json').read_text())
    run = HERE / 'runs' / manifest['branch_run_id']
    state = json.loads((run / 'state.json').read_text())
    current = int(state['round'])
    require(50 <= current <= 100, 'R50 smoke or later not committed')
    rounds = [json.loads(line) for line in (run / 'rounds.jsonl').read_text().splitlines()]
    require(len(rounds) >= current, 'committed history was truncated')
    rounds = rounds[:current]
    require([row['round'] for row in rounds] == list(range(1, current + 1)),
            'round records are not contiguous')
    require(rounds[48]['checkpoint_sha256'] == manifest['source_files_sha256']['adapter_49.pt'],
            'R49 branch source checkpoint changed')
    schedule = {row['round']: row for row in manifest['factor_rounds']}
    count = {'normal': 0, 'contention': 0, 'recovery': 0}
    nx_total = sum(row['clients']['nx']['samples'] for row in rounds[:49])
    for rnd in range(50, current + 1):
        row = rounds[rnd - 1]
        frozen = schedule[rnd]
        require(row['global_start_checkpoint_sha256'] == rounds[rnd - 2]['checkpoint_sha256'],
                f'R{rnd} checkpoint chain broken')
        require(row['phase'] == frozen['phase'] and
                row['nano_compute_seconds_per_token_factor'] == frozen['nano_compute_seconds_per_token_factor'] and
                row['fig9_reference_round'] == frozen['fig9_reference_round'],
                f'R{rnd} phase or measured factor changed')
        pred = row['pre_round_prediction']
        require(pred['postwarmup_window_gate'] == 'PASS' and
                pred['native_fallback_reason'] is None and
                pred['nx_final_s'] <= pred['allowed_s'] + 1e-9,
                f'R{rnd} prediction window gate failed')
        total = sum(client['samples'] for client in row['clients'].values())
        require(total > 0 and math.isclose(sum(row['aggregation_weights'].values()), 1.0,
                                           abs_tol=1e-9), f'R{rnd} aggregate invalid')
        for name, client in row['clients'].items():
            require(client['samples'] == client['optimizer_steps'] == len(client['visited_sample_ids']),
                    f'R{rnd}/{name} actual sample accounting changed')
            require(math.isclose(row['aggregation_weights'][name], client['samples'] / total,
                                 abs_tol=1e-9), f'R{rnd}/{name} aggregation weight changed')
            if client['samples']:
                require(client['local_training_start_checkpoint_sha256'] ==
                        row['global_start_checkpoint_sha256'],
                        f'R{rnd}/{name} trained from another checkpoint')
        nx_total += row['clients']['nx']['samples']
        require(nx_total <= 2 * 18441 and
                math.isclose(row['nx_cumulative_exposure'], nx_total / 18441,
                             abs_tol=1e-12), f'R{rnd} NX exposure accounting changed')
        nano_projection = row['edge_compute_projection']['nano']
        factor = frozen['nano_compute_seconds_per_token_factor']
        if row['clients']['nano']['samples']:
            require(math.isclose(nano_projection['compute_s'],
                                 nano_projection['normal_profile_compute_s'] * factor,
                                 rel_tol=1e-12), f'R{rnd} Nano factor was not applied after training')
            require((nano_projection['compute_energy_j'] is None) ==
                    (row['phase'] != 'normal'), f'R{rnd} Nano energy support mislabeled')
        snapshot = run / 'native_flower_log/resume_state' / f'round_{rnd:02d}.pkl'
        require(sha(snapshot) == row['native_strategy_state_sha256'],
                f'R{rnd} native Flower snapshot hash changed')
        count[row['phase']] += 1
    require(sha(run / f'adapter_{current}.pt') == state['checkpoint_sha256'] ==
            rounds[-1]['checkpoint_sha256'], 'current adapter checkpoint hash changed')
    require(state['native_strategy_state_sha256'] == rounds[-1]['native_strategy_state_sha256'],
            'current native state hash changed')
    require(state['run_id'] == manifest['branch_run_id'] and
            state['protocol_sha256'] == EXPECTED_MANIFEST_SHA,
            'branch state identity changed')
    evaluations = [json.loads(line) for line in (run / 'validation.jsonl').read_text().splitlines()]
    expected_evaluations = [r for r in range(0, current + 1, 10)]
    require([row['round'] for row in evaluations[:len(expected_evaluations)]] == expected_evaluations,
            'required validation checkpoints are missing')
    for row in evaluations[:len(expected_evaluations)]:
        require(sha(run / f'adapter_{row["round"]}.pt') == row['checkpoint_sha256'],
                f'R{row["round"]} evaluation checkpoint hash changed')
    print(json.dumps({'status': 'PASS', 'committed_round': current,
                      'audited_branch_rounds': current - 49, 'phase_counts': count,
                      'nx_executed_samples': nx_total,
                      'nx_exposure': nx_total / 18441,
                      'checkpoint_sha256': state['checkpoint_sha256'],
                      'native_state_sha256': state['native_strategy_state_sha256']},
                     indent=2))


if __name__ == '__main__':
    main()
