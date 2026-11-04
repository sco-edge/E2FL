"""Run on a Flower 1.22 host with source snapshot on PYTHONPATH; no training."""
import copy, json, os, tempfile
from pathlib import Path
import torch
from flwr.app import ArrayRecord, ConfigRecord
from e2fl.strategy.FedAvgLoRA import FedAvgLoRA
from e2fl.strategy.LoRACastCommAware import LoRACastCommAware

class Grid:
    def get_node_ids(self): return [11,22]

with tempfile.TemporaryDirectory() as d:
    os.environ['TTA_OUTPUT_DIR']=d
    s=LoRACastCommAware(ROOT=d,log_dir='logs',min_available_nodes=2,
                       enable_rpi_only_rounds=False,use_token_ratio=False,tokens_per_example=64)
    s.cid_to_device={11:'RPi5',22:'jetson_orin_nx'}
    s.cid_to_device_uid={11:'rpi',22:'jetson'}
    for r in [1,2]:
        s.round_stats[r]={dev:dict(tokens=128,tokens_per_client=128,train_time=t,num_examples=2,num_examples_per_client=2) for dev,t in [('RPi5',10),('jetson_orin_nx',2)]}
    baseline=copy.deepcopy(s)
    cfg=ConfigRecord({'batch_size':1,'local_epochs':1,'max_samples_per_round':2,'max_seq_len':128,'rpi-normal-quota':2,'jetson-base-quota':2,'jetson-max-quota':16,'jetson-sample-step':2})
    arrays=ArrayRecord({'test':torch.ones(1)})
    original=FedAvgLoRA.configure_train
    expected=list(original(baseline,3,arrays,cfg,Grid()))
    from instrumentation import install
    install()
    actual=list(s.configure_train(3,arrays,cfg,Grid()))
    assert len(actual)==len(expected)==2
    for key in ['round_stats','time_scale','beta_sec_per_token','round_configs']:
        assert getattr(s,key)==getattr(baseline,key),key
    rows=[json.loads(x) for x in (Path(d)/'predictions.jsonl').read_text().splitlines()]
    assert all(x['pred_train_time_s']>0 and x['history_rounds']==[1,2] for x in rows)
    # Parent's diagnostic signature remains usable on communication-aware subclass.
    s._predict_comm_time('RPi5:rpi','RPi5',100,0)
    print(json.dumps({'audit_state_unchanged':True,'dispatch_count':len(actual),'predictions':[{k:x[k] for k in ['round','device_name','pred_train_time_s','history_rounds']} for x in rows]}))
