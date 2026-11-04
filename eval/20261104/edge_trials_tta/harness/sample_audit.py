"""Observation-only sample and local-loop timestamps for the isolated TTA app.

The wrapper delegates indexing to torch Subset unchanged. It records only a
partition-local row number and a SHA-256 of token IDs, never sample text.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from torch.utils.data import Subset

OUT = Path(os.environ['TTA_OUTPUT_DIR'])


def _append(name, row):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def event(round_number, device_name, device_uid, partition_id, kind, workload=None):
    _append('training_events.jsonl', {
        'round': int(round_number), 'device_name': str(device_name),
        'device_uid': str(device_uid), 'partition_id': int(partition_id),
        'kind': kind, 'unix_time_s': time.time(), 'workload':workload,
    })


def adapter_modules(model):
    """Read the active PEFT target modules without changing the model."""
    try:
        configs=getattr(model,'peft_config',None)
        if not configs:
            return None
        modules=next(iter(configs.values())).target_modules
        return sorted(str(item) for item in modules) if modules else None
    except (AttributeError,StopIteration,TypeError):
        return None


def _token_fields(item):
    tokens = item.get('input_ids') if isinstance(item, dict) else None
    if hasattr(tokens, 'tolist'):
        tokens = tokens.tolist()
    if not isinstance(tokens, list) or not tokens:
        raise ValueError('TTA sample has no tokenized input_ids')
    return {
        'token_ids_sha256': hashlib.sha256(
            json.dumps(tokens, separators=(',', ':')).encode()
        ).hexdigest(),
        'input_token_count': len(tokens),
    }


def record_planned_window(dataset, partition_id, device_name, device_uid,
                          model, dataset_name, offset, count):
    """Freeze input lengths of future evaluation rows without training them."""
    if offset < 0 or count < 1 or offset + count > len(dataset):
        raise ValueError('TTA planned sample window exceeds data partition')
    path = OUT / 'planned_samples.jsonl'
    if path.is_file():
        existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if (len(existing) == count
                and {x.get('partition_train_row') for x in existing} == set(range(offset,offset+count))
                and all(x.get('partition_id') == partition_id
                        and x.get('device_name') == device_name
                        and x.get('device_uid') == device_uid
                        and x.get('model') == model
                        and x.get('dataset') == dataset_name for x in existing)):
            return
        raise ValueError('TTA planned sample manifest changed within the attempt')
    rows = []
    for row_number in range(offset, offset + count):
        rows.append({
            'partition_id': int(partition_id), 'device_name': str(device_name),
            'device_uid': str(device_uid), 'model': str(model),
            'dataset': str(dataset_name), 'partition_train_row': row_number,
            'recorded_at': time.time(), **_token_fields(dataset[row_number]),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class AuditSubset(Subset):
    def __init__(self, dataset, indices, round_number, device_name, device_uid,
                 partition_id):
        super().__init__(dataset, indices)
        self.audit = {'round': int(round_number), 'device_name': str(device_name),
                      'device_uid': str(device_uid), 'partition_id': int(partition_id)}

    def _record(self, local_index, item):
        _append('sample_audit.jsonl', {
            **self.audit, 'partition_train_row': int(self.indices[local_index]),
            **_token_fields(item),
            'read_at': time.time(),
        })

    def __getitem__(self, index):
        item = super().__getitem__(index)
        self._record(index, item)
        return item

    def __getitems__(self, indices):
        parent = super()
        items = (parent.__getitems__(indices) if hasattr(parent, '__getitems__')
                 else [parent.__getitem__(index) for index in indices])
        for index, item in zip(indices, items):
            self._record(index, item)
        return items
