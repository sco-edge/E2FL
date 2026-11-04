"""JSON-only predictor checkpoint for the isolated three-round TTA trial."""
import json
import math
import os
from pathlib import Path

GROUPS = ('RPi5', 'jetson_orin_nx')
TABLES = ('beta_sec_per_token', 'time_scale', 'upload_rate_bps',
          'upload_rate_Bps', 'upload_bytes')

def _finite(value):
    if isinstance(value, dict):
        return all(isinstance(k, str) and _finite(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    return isinstance(value, (int, float)) and math.isfinite(value)

def validate(state):
    if state.get('schema') != 1 or state.get('source_rounds') != [1, 2]:
        raise ValueError('predictor state must contain exactly two priming rounds')
    stats = state.get('round_stats')
    if not isinstance(stats, dict) or set(stats) != {'1', '2'}:
        raise ValueError('priming round statistics are incomplete')
    for round_stats in stats.values():
        for group in GROUPS:
            entry = round_stats.get(group, {})
            if float(entry.get('tokens', 0)) <= 0 or float(entry.get('train_time', 0)) <= 0:
                raise ValueError(f'missing positive observations for {group}')
    if not isinstance(state.get('window_history'), list) or len(state['window_history']) != 2:
        raise ValueError('priming participation history is incomplete')
    for name in TABLES:
        if not isinstance(state.get(name), dict):
            raise ValueError(f'missing predictor table {name}')
    if not _finite(state):
        raise ValueError('nonfinite predictor state')
    return state

def capture(strategy):
    if set(strategy.round_stats) != {1, 2}:
        raise ValueError('prime capture requires exactly rounds 1 and 2')
    state = {'schema': 1, 'source_rounds': [1, 2],
             'round_stats': {str(k): v for k, v in strategy.round_stats.items()},
             'window_history': strategy.window_history[-2:],
             'slowest_device_name': strategy.slowest_device_name}
    for name in TABLES:
        state[name] = dict(getattr(strategy, name, {}))
    return validate(state)

def write(path, state):
    path = Path(path)
    body = json.dumps(validate(state), sort_keys=True, allow_nan=False)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def restore(strategy, state):
    validate(state)
    if strategy.round_stats:
        raise ValueError('predictor state can only be restored into a fresh strategy')
    # Two historical observations precede live round 1. No Flower node IDs,
    # model weights, round configs or checkpoint paths cross this boundary.
    strategy.round_stats = {-1: state['round_stats']['1'],
                             0: state['round_stats']['2']}
    strategy.min_beta_round = -1
    strategy.window_history = list(state['window_history'])
    strategy.slowest_device_name = state.get('slowest_device_name')
    for name in TABLES:
        if hasattr(strategy, name):
            setattr(strategy, name, dict(state[name]))
    return strategy

def read(path):
    return validate(json.loads(Path(path).read_text(),
                               parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))))
