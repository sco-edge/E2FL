#!/usr/bin/env python3
"""Write the final full-checkpoint hash audit for the isolated R50–R100 run.

This script only writes final_integrity.json in the isolated branch directory.
It can run only after R100 is fully committed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import audit_progress


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'experiment_manifest.json'
PARENT_V2 = HERE.parent / 'lower_floor_v2'


def require(ok: bool, description: str) -> None:
    if not ok:
        raise ValueError(description)


def record(path: Path) -> dict:
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': audit_progress.sha(path)}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    run = HERE / 'runs' / manifest['branch_run_id']
    state = json.loads((run / 'state.json').read_text())
    require(state['round'] == 100, 'R100 is not fully committed')
    audit_progress.main()
    rounds = [json.loads(line) for line in (run / 'rounds.jsonl').read_text().splitlines()]
    require(len(rounds) == 100 and [r['round'] for r in rounds] == list(range(1, 101)),
            'round history is not exactly R1–R100')

    source = Path(manifest['source_directory'])
    parent_files = {}
    for name, expected in sorted(manifest['source_files_sha256'].items()):
        entry = record(source / name)
        require(entry['sha256'] == expected, f'parent source changed: {name}')
        parent_files[name] = entry
    parent_provenance = {
        'manifest': record(PARENT_V2 / 'nano_lower_floor_experiment_manifest_v2.json'),
        'run_adaptive_training': record(PARENT_V2 / 'nano_lower_floor_rtx_runner_v2.py'),
        'stress_snapshot': record(PARENT_V2 / 'source_stress_manifest_snapshot.json'),
    }
    require(parent_provenance['manifest']['sha256'] == manifest['source_manifest_sha256'],
            'parent manifest changed')
    require(parent_provenance['run_adaptive_training']['sha256'] == manifest['source_runner_sha256'],
            'parent run_adaptive_training changed')
    require(parent_provenance['stress_snapshot']['sha256'] ==
            manifest['source_stress_snapshot_sha256'], 'parent stress snapshot changed')

    checkpoints = {}
    for rnd in [0, 10, 20, 30, 40, *range(49, 101)]:
        path = run / f'adapter_{rnd}.pt'
        entry = record(path)
        expected = (manifest['source_files_sha256'][path.name] if rnd <= 49 else
                    rounds[rnd - 1]['checkpoint_sha256'])
        require(entry['sha256'] == expected, f'branch adapter R{rnd} changed')
        checkpoints[str(rnd)] = entry

    native_snapshots = {}
    for rnd in range(49, 101):
        base = run / 'native_flower_log' / 'resume_state'
        pkl = base / f'round_{rnd:02d}.pkl'
        metadata_path = base / f'round_{rnd:02d}.json'
        entry = record(pkl)
        metadata = json.loads(metadata_path.read_text())
        metadata_entry = record(metadata_path)
        expected = (metadata['state_sha256'] if rnd == 49 else
                    rounds[rnd - 1]['native_strategy_state_sha256'])
        require(entry['sha256'] == expected == metadata['state_sha256'],
                f'branch native snapshot R{rnd} changed')
        require(metadata['completed_round'] == rnd and
                metadata['experiment_id'] == manifest['branch_run_id'] and
                metadata['checkpoint'] == str(run / f'adapter_{rnd}.pt') and
                metadata['checkpoint_sha256'] == checkpoints[str(rnd)]['sha256'] and
                metadata['state_file'] == pkl.name,
                f'branch native metadata R{rnd} changed')
        native_snapshots[str(rnd)] = {'pickle': entry, 'metadata': metadata_entry}

    validations = [json.loads(line) for line in (run / 'validation.jsonl').read_text().splitlines()]
    require([v['round'] for v in validations] == list(range(0, 101, 10)),
            'validation history does not contain exactly R0,10,...,100')
    for v in validations:
        require(v['checkpoint_sha256'] == checkpoints[str(v['round'])]['sha256'],
                f'validation checkpoint R{v["round"]} changed')

    report = {
        'status': 'PASS',
        'run_id': manifest['branch_run_id'],
        'round': 100,
        'branch_manifest': record(MANIFEST),
        'branch_runner': record(HERE / 'branch_run_adaptive_training.py'),
        'parent_source_files': parent_files,
        'parent_provenance': parent_provenance,
        'checkpoints': checkpoints,
        'native_snapshots': native_snapshots,
        'logs': {name: record(run / name) for name in
                 ('rounds.jsonl', 'validation.jsonl', 'state.json')},
        'checkpoint_count': len(checkpoints),
        'native_snapshot_count': len(native_snapshots),
        'validation_count': len(validations),
    }
    destination = HERE / 'final_integrity.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, destination)
    print(json.dumps({'status': report['status'], 'round': 100,
                      'checkpoint_count': report['checkpoint_count'],
                      'native_snapshot_count': report['native_snapshot_count'],
                      'validation_count': report['validation_count'],
                      'report_path': str(destination),
                      'report_sha256': audit_progress.sha(destination)}, indent=2))


if __name__ == '__main__':
    main()
