"""Isolated Flower strategy adapter for the exposure-constrained proposal.

This experimental path inherits the original estimator and S/H allocator.
It deliberately disables participation-based rebalancing. The verified edge
profile only supports E=1 here; E>1 candidates require separately measured
cost predictions and are rejected rather than silently projected.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from e2fl.strategy.LoRACast import LoRACastCommAware
from exposure_controller import Candidate, Policy, choose_workload
from policy_resume import run_remaining_rounds


class FinalExposureLoRACast(LoRACastCommAware):
    def __init__(self, *, exposure_max: float | None, nx_eligible_unique: int,
                 nx_minimum_samples: int = 0, nx_maximum_samples: int = 0,
                 nx_maximum_epochs: int = 1, **kwargs):
        if (nx_eligible_unique <= 0 or nx_minimum_samples < 0 or nx_maximum_samples < 0
                or nx_maximum_samples and nx_minimum_samples > nx_maximum_samples):
            raise ValueError('invalid NX dataset/workload policy')
        if exposure_max is not None and (not math.isfinite(exposure_max) or exposure_max < 0):
            raise ValueError('invalid cumulative exposure cap')
        if nx_maximum_epochs != 1:
            raise ValueError('E>1 lacks a validated edge cost profile in this adapter')
        kwargs['rebalancing_enabled'] = False
        kwargs['enable_rpi_only_rounds'] = True  # permits pretraining message omission
        super().__init__(**kwargs)
        self.exposure_max = exposure_max
        self.nx_eligible_unique = int(nx_eligible_unique)
        # Zero delegates the bound to the original allocator's own profile.
        self.nx_minimum_samples = int(nx_minimum_samples)
        self.nx_maximum_samples = int(nx_maximum_samples)
        self.nx_maximum_epochs = int(nx_maximum_epochs)
        self.nx_cumulative_samples = 0
        self.nx_unique_visited = 0
        self.nx_seen_ids = set()
        self.exposure_decisions = {}
        self.availability_events = []
        self.resume_completed_round = 0  # activates existing model/state checkpoint hook

    def start(self, *args, **kwargs):
        if not self.resume_completed_round:
            return super().start(*args, **kwargs)
        if args:
            raise ValueError('round-boundary recovery requires keyword arguments')
        return run_remaining_rounds(
            self, completed_round=self.resume_completed_round, **kwargs
        )

    def announce_new_data(self, *, current_round: int, eligible_unique: int,
                          training_manifest_sha256: str) -> None:
        """Accept only a current-round availability event, never future data."""
        if current_round < 1 or eligible_unique < self.nx_eligible_unique:
            raise ValueError('invalid data arrival')
        if len(training_manifest_sha256) != 64:
            raise ValueError('data arrival requires a SHA-256 training manifest')
        if current_round in self.exposure_decisions:
            raise ValueError('arrival came after round decision')
        self.nx_eligible_unique = int(eligible_unique)
        self.availability_events.append({'round':int(current_round),
            'eligible_unique':int(eligible_unique),'manifest_sha256':training_manifest_sha256})

    def update_train_config(self, server_round, train_config):
        config=super().update_train_config(server_round,train_config)
        if str(config.get('round_mode','BOTH')) != 'BOTH':
            raise RuntimeError('final controller unexpectedly invoked old rebalancing')
        normal_s=int(config.get('max_samples_per_round_jetson_orin_nx',0) or 0)
        normal_h=int(config.get('local_steps_jetson_orin_nx',0) or 0)
        normal_e=int(config.get('local_epochs_jetson_orin_nx',1) or 1)
        if normal_e != 1 or normal_s < 0:
            raise RuntimeError('unexpected original allocator S/E semantics')
        minimum=max(int(self.jetson_base_quota),self.nx_minimum_samples)
        maximum=min(int(self.jetson_max_quota),
                    self.nx_maximum_samples if self.nx_maximum_samples else int(self.jetson_max_quota))
        if maximum<minimum:
            raise RuntimeError('user NX ceiling lies below the native minimum candidate')
        batch=int(config['batch_size'])
        sequence_cap=int(config.get('max_seq_len',64) or 64)
        normal_work=Candidate(normal_s,1,normal_h,batch,sequence_cap,0.0).executed()[1]
        cap_total=(math.floor(self.exposure_max*self.nx_eligible_unique+1e-9)
                   if self.exposure_max is not None else None)
        remaining=cap_total-self.nx_cumulative_samples if cap_total is not None else None
        if normal_s<=min(maximum,self.nx_eligible_unique) and (remaining is None or remaining>=normal_work):
            config['exposure_tracking_enabled']=True
            self.exposure_decisions[server_round]={'normal_s':normal_s,'selected_s':normal_s,
                'action':'normal','remaining_before':remaining,'eligible_unique':self.nx_eligible_unique}
            return config
        # Binding budget: evaluate only candidates in the original S lattice.
        pi_s=int(config.get('max_samples_per_round_RPi5',0) or 0)
        reference=self._predict_train_time('RPi5',float(pi_s),config)
        uncalibrated=reference<=0
        if uncalibrated:
            # R1 warm-up can lack a calibrated time estimate. Only reduce the
            # original native quota, using S units for a monotonic safety
            # comparison. This is not a seconds-valued cost prediction.
            reference=float(max(1,normal_s))
        def make_candidate(s):
            candidate_cfg=copy.deepcopy(config)
            candidate_cfg['max_samples_per_round_jetson_orin_nx']=int(s)
            candidate_cfg['local_epochs_jetson_orin_nx']=1
            candidate_cfg['local_steps_jetson_orin_nx']=normal_h
            predicted=(float(s) if uncalibrated else
                       self._predict_train_time('jetson_orin_nx',float(s),candidate_cfg))
            if predicted<=0:predicted=math.inf
            return Candidate(int(s),1,normal_h,int(config['batch_size']),
                int(config.get('max_seq_len',64) or 64),float(predicted))
        normal=make_candidate(normal_s)
        lattice=[make_candidate(s) for s in range(minimum,
            min(normal_s,maximum,self.nx_eligible_unique)+1,int(self.jetson_sample_step))]
        policy=Policy(maximum,minimum,1,0,batch,sequence_cap,self.exposure_max)
        decision=choose_workload(normal=normal,alternatives=lattice,policy=policy,
            eligible_unique_samples=self.nx_eligible_unique,
            cumulative_executed=self.nx_cumulative_samples,
            previously_visited_unique=self.nx_unique_visited,
            compute_window_s=reference*float(self.jetson_target_margin))
        selected=decision.selected.samples if decision.selected else 0
        config['max_samples_per_round_jetson_orin_nx']=int(selected)
        config['local_epochs_jetson_orin_nx']=1 if selected else 0
        config['local_steps_jetson_orin_nx']=normal_h if selected else 0
        for uid,device in self._known_client_device_map().items():
            if device=='jetson_orin_nx':
                suffix=self._safe_config_suffix(uid)
                config[f'max_samples_per_round_uid_{suffix}']=int(selected)
                config[f'local_epochs_uid_{suffix}']=1 if selected else 0
                config[f'local_steps_uid_{suffix}']=normal_h if selected else 0
        if selected==0:
            protected={uid for uid,device in self._known_client_device_map().items() if device=='RPi5'}
            if not protected:
                raise RuntimeError('cannot safely omit NX before Pi Flower destination is known')
            self.protected_client_ids=protected
            config['round_mode']='PROTECTED_ONLY'
        self.exposure_decisions[server_round]={'normal_s':normal_s,'selected_s':selected,
            'action':decision.action,'remaining_before':remaining,
            'native_minimum_s':minimum,'native_sample_step':int(self.jetson_sample_step),
            'user_maximum_s':maximum,'expected_processed_samples':decision.actual_processed_samples,
            'eligible_unique':self.nx_eligible_unique,
            'candidate_cost_status':('uncalibrated_monotonic_S_proxy' if uncalibrated else
                                     'native_estimator_prediction'),
            'predicted_residual_window_s':(None if uncalibrated else
                                           decision.predicted_residual_window_s)}
        path=Path(self.log_dir)/f'final_exposure_decision_round_{server_round:03d}.json'
        config['exposure_tracking_enabled']=True
        path.write_text(json.dumps(self.exposure_decisions[server_round],indent=2)+'\n')
        return config

    def aggregate_train(self, server_round, replies):
        replies=list(replies)
        observed=0
        unique=set()
        for msg in replies:
            if not msg.has_content() or 'metrics' not in msg.content:continue
            m=dict(msg.content['metrics'].items())
            name=self._decode_metric_string(m.get('device_name_chars'))
            if name!='jetson_orin_nx':continue
            if int(m.get('eligible_unique_samples',-1)) != self.nx_eligible_unique:
                raise RuntimeError('client/server eligible-data size mismatch')
            observed+=int(m.get('num-examples',0))
            ids=m.get('visited_sample_ids',[])
            if len(ids)!=int(m.get('num-examples',0)):
                raise RuntimeError('client did not report actual visited samples')
            unique.update(int(x) for x in ids)
        expected=int(self.exposure_decisions[server_round]['selected_s'])
        if observed>expected:
            raise RuntimeError('executed samples exceed scheduled exposure quota')
        if self.exposure_max is not None and self.nx_cumulative_samples+observed > math.floor(
                self.exposure_max*self.nx_eligible_unique+1e-9):
            raise RuntimeError('cumulative exposure violated; refusing aggregation')
        # Validate the work before the inherited method aggregates or records a
        # global state. A rejected client report must not advance the model.
        arrays,metrics=super().aggregate_train(server_round,replies)
        if arrays is None:return arrays,metrics
        self.nx_cumulative_samples+=observed
        self.nx_seen_ids.update(unique)
        self.nx_unique_visited=len(self.nx_seen_ids)
        return arrays,metrics
