"""Opt-in TTA audit and per-round test configuration for an isolated app.

The estimator equation and learned coefficients are unchanged. For the fixed
TTA workload, the token-count input comes from a frozen pre-training manifest.
"""
import copy, json, os, time
from pathlib import Path
from run_settings import (WARMUP_ROUNDS, WARMUP_QUOTA,
    WARMUP_JETSON_QUOTA, STATE_MODE, PLANNED_TOKENS_PER_EXAMPLE)

ROOT = Path(os.environ['TTA_OUTPUT_DIR'])
ROOT.mkdir(parents=True, exist_ok=True)
SETTINGS = {'warmup_rounds': WARMUP_ROUNDS, 'warmup_quota': WARMUP_QUOTA,
            'warmup_jetson_quota': WARMUP_JETSON_QUOTA}

def emit(event, filename='predictions.jsonl'):
    with (ROOT / filename).open('a') as f:
        f.write(json.dumps(event, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())

def model_state(predictor, key, device, cfg, examples):
    """Describe the exact arguments and model state used by a forecast copy."""
    used_key = key
    beta = predictor.beta_sec_per_token.get(key)
    if beta is None and key != device:
        beta = predictor.beta_sec_per_token.get(device)
        used_key = device
    if beta is None:
        beta = predictor._estimate_beta_from_history(key, cfg)
        if beta is None and key != device:
            beta = predictor._estimate_beta_from_history(device, cfg)
            used_key = device
    sample_count, epochs = predictor._resolve_effective_workload(key, device, cfg, examples)
    steps = predictor._resolve_effective_steps(key, device, cfg)
    tokens_per_example = predictor._estimate_tokens_per_example(key, device)
    tokens_per_step = (predictor._estimate_tokens_per_step_for_device(device, cfg)
                       if steps > 0 else None)
    token_workload = min(sample_count * epochs * tokens_per_example,
                         steps * tokens_per_step) if steps > 0 else sample_count * epochs * tokens_per_example
    alpha = predictor.time_scale.get(key, predictor.time_scale.get(device, 1.0))
    return dict(identity_key=key, beta_source_key=used_key, beta_sec_per_token=beta,
                alpha=float(alpha), effective_samples=sample_count,
                effective_epochs=epochs, effective_steps=steps,
                tokens_per_example=tokens_per_example,
                tokens_per_step=tokens_per_step, token_workload=token_workload)

def warmup_config(server_round, config, settings=None):
    """Keep the first test rounds short without changing measured-round policy."""
    settings = SETTINGS if settings is None else settings
    rounds = int(settings.get('warmup_rounds', 0) or 0)
    quota = int(settings.get('warmup_quota', 0) or 0)
    jetson_quota = int(settings.get('warmup_jetson_quota', quota) or quota)
    if quota <= 0 or server_round > rounds:
        return config
    adjusted = copy.deepcopy(config)
    adjusted['max_samples_per_round'] = max(quota, jetson_quota)
    for key in ('rpi-normal-quota', 'rpi-correction-max-quota'):
        adjusted[key] = quota
    for key in ('jetson-base-quota', 'jetson-max-quota'):
        adjusted[key] = jetson_quota
    return adjusted

def install():
    from e2fl.strategy.FedAvgLoRA import FedAvgLoRA
    if PLANNED_TOKENS_PER_EXAMPLE:
        if STATE_MODE != 'reuse' or set(PLANNED_TOKENS_PER_EXAMPLE) != {'RPi5','jetson_orin_nx'}:
            raise RuntimeError('TTA planned token inputs are incomplete or in the wrong state mode')
        historical_tokens_per_example = FedAvgLoRA._estimate_tokens_per_example
        def planned_tokens_per_example(self, identity_key, device_name):
            if device_name not in PLANNED_TOKENS_PER_EXAMPLE:
                return historical_tokens_per_example(self, identity_key, device_name)
            value = float(PLANNED_TOKENS_PER_EXAMPLE[device_name])
            if value <= 0:
                raise RuntimeError('TTA planned token input must be positive')
            return value
        # Replace only the source of the workload input in the disposable app.
        # beta, alpha, the estimator equation and allocation policy are untouched.
        FedAvgLoRA._estimate_tokens_per_example = planned_tokens_per_example
    original = FedAvgLoRA.configure_train
    def configure(self, server_round, arrays, config, grid):
        messages = list(original(self, server_round, arrays,
                                 warmup_config(server_round, config), grid))
        cfg = self.round_configs[server_round]
        # The predictor caches beta; deepcopy prevents audit calls modifying policy state.
        predictor = copy.deepcopy(self)
        for msg in messages:
            cid = msg.metadata.dst_node_id
            dev = self.cid_to_device.get(cid, self.cid_to_device.get(str(cid)))
            uid = self.cid_to_device_uid.get(cid, self.cid_to_device_uid.get(str(cid), str(cid)))
            pred = None
            candidates = {}
            states = {}
            if dev:
                key = f'{dev}:{uid}'
                examples = float(cfg.get('max_samples_per_round', 0))
                pred = FedAvgLoRA._predict_train_time(predictor, key, examples, cfg)
                if pred <= 0:
                    pred = FedAvgLoRA._predict_train_time(predictor, dev, examples, cfg)
                    key = dev
                states[dev] = model_state(predictor, key, dev, cfg, examples)
            elif STATE_MODE == 'reuse':
                # The new Flower node IDs have no device mapping until their
                # first replies. Commit both group forecasts before dispatch;
                # the reply's CID selects the corresponding precommitted value.
                for group in ('RPi5', 'jetson_orin_nx'):
                    key = 'max_samples_per_round_'+group
                    examples = float(cfg.get(key, cfg.get('max_samples_per_round', 0)))
                    candidates[group] = FedAvgLoRA._predict_train_time(
                        predictor, group, examples, cfg)
                    states[group] = model_state(predictor, group, group, cfg, examples)
            emit(dict(round=server_round, cid=str(cid), device_name=dev, device_uid=str(uid),
                      predicted_at=time.time(), pred_train_time_s=pred,
                      candidates=candidates, model_states=states,
                      coefficient_last_updated_round=max(self.round_stats, default=None),
                      history_rounds=sorted(self.round_stats), config=dict(cfg)))
        return messages
    FedAvgLoRA.configure_train = configure
    original_evaluate = FedAvgLoRA.aggregate_evaluate
    def aggregate_evaluate(self, server_round, *args, **kwargs):
        result = original_evaluate(self, server_round, *args, **kwargs)
        emit(dict(round=server_round, ended_at=time.time()), 'round_boundaries.jsonl')
        return result
    FedAvgLoRA.aggregate_evaluate = aggregate_evaluate
    if STATE_MODE == 'prime':
        from predictor_state import capture, write
        aggregate_original = FedAvgLoRA.aggregate_train
        def aggregate(self, server_round, replies):
            result = aggregate_original(self, server_round, replies)
            if server_round == 2:
                write(ROOT/'predictor_state.json', capture(self))
            return result
        FedAvgLoRA.aggregate_train = aggregate
    # Existing subclass's one-argument helper collides with parent logging's
    # four-argument helper. Adapt logging calls only; policy calls are untouched.
    from e2fl.strategy.LoRACastCommAware import LoRACastCommAware
    old_comm = LoRACastCommAware._predict_comm_time
    def comm(self, identity, *args):
        return FedAvgLoRA._predict_comm_time(self, identity, *args) if args else old_comm(self, identity)
    LoRACastCommAware._predict_comm_time = comm
