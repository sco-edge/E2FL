"""Observation-only forecasts recorded before Flower dispatch in copied apps."""
import copy
import json
from pathlib import Path
import time


def install(strategy):
    strategy_class=type(strategy)
    original=strategy_class.configure_train
    def configure(self, server_round, arrays, config, grid):
        strategy=self
        messages=list(original(self,server_round,arrays,config,grid))
        predictor=copy.deepcopy(strategy)
        cfg=predictor.round_configs[server_round]
        records=[]
        for device in ('RPi5','jetson_orin_nx'):
            samples=float(cfg.get('max_samples_per_round_'+device,cfg.get('max_samples_per_round',0)))
            group_prediction=predictor._predict_train_time(device,samples,cfg)
            keys=[]
            for msg in messages:
                cid=msg.metadata.dst_node_id
                dev=predictor.cid_to_device.get(cid,predictor.cid_to_device.get(str(cid)))
                if dev==device:
                    uid=predictor.cid_to_device_uid.get(cid,predictor.cid_to_device_uid.get(str(cid),str(cid)))
                    keys.append((str(cid),f'{device}:{uid}'))
            if not keys:keys=[('',device)]
            for cid,key in keys:
                predicted=predictor._predict_train_time(key,samples,cfg)
                records.append(dict(round=server_round,device_name=device,cid=cid,
                    identity_key=key,predicted_at=time.time(),predicted_s=predicted,
                    group_predicted_s=group_prediction,requested_samples=samples,
                    estimated_tokens_per_example=predictor._estimate_tokens_per_example(key,device),
                    history_rounds=sorted(predictor.round_stats),
                    prediction_source='before_dispatch',config=dict(cfg)))
        path=Path(strategy.log_dir)/'policy_predictions.jsonl'
        with path.open('a') as stream:
            for record in records:stream.write(json.dumps(record,allow_nan=False)+'\n')
            stream.flush()
        return messages
    # Resume snapshots serialize instance state. Keep instrumentation on the
    # class so no local function enters the checkpoint payload.
    strategy_class.configure_train=configure
