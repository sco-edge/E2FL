"""Control-path tests against Flower message construction and omission filtering."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, RecordDict

from e2fl.strategy.FinalExposureLoRACast import FinalExposureLoRACast
from e2fl.strategy.LoRACast import LoRACastCommAware
from e2fl.server_app_loracast import create_strategy
from policy_resume import load_completed_round, save_completed_round


class TwoNodeGrid:
    def get_node_ids(self):
        return [1,2]


class FlowerWiringTest(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def make_strategy(self,cap,cumulative):
        s=FinalExposureLoRACast(exposure_max=cap,nx_eligible_unique=128,
            fraction_train=1.0,fraction_evaluate=0.0,min_available_nodes=2,
            log_dir='isolated',ROOT=str(self.root),experiment_id='',
            aggregation_mode='processed_samples')
        s.nx_cumulative_samples=cumulative
        # This isolated measured-profile fixture has native S8,12,...,32.
        s.jetson_base_quota=8
        s.jetson_sample_step=4
        s.jetson_max_quota=32
        s.cid_to_device_uid={1:'pi',2:'nx','1':'pi','2':'nx'}
        s._known_client_device_map=lambda:{'pi':'RPi5','nx':'jetson_orin_nx'}
        s._predict_train_time=lambda name,samples,config: 32.0 if name=='RPi5' else float(samples)
        return s

    def configure(self,strategy):
        config=ConfigRecord({'batch_size':1,'max_seq_len':64,'round_mode':'BOTH',
            'max_samples_per_round_RPi5':4,'local_epochs_RPi5':1,'local_steps_RPi5':0,
            'max_samples_per_round_jetson_orin_nx':32,
            'local_epochs_jetson_orin_nx':1,'local_steps_jetson_orin_nx':0})
        arrays=ArrayRecord(torch_state_dict=OrderedDict({'adapter':torch.tensor([0.])}))
        with patch.object(LoRACastCommAware,'update_train_config',lambda self,r,c:c):
            messages=list(strategy.configure_train(3,arrays,config,TwoNodeGrid()))
        return messages,strategy.round_configs[3]

    def test_nonbinding_follows_original_two_client_flower_path(self):
        s=self.make_strategy(None,0)
        messages,config=self.configure(s)
        self.assertEqual({m.metadata.dst_node_id for m in messages},{1,2})
        self.assertEqual(config['max_samples_per_round_jetson_orin_nx'],32)

    def test_partial_quota_is_sent_to_both_clients(self):
        s=self.make_strategy(2,232)
        messages,config=self.configure(s)
        self.assertEqual({m.metadata.dst_node_id for m in messages},{1,2})
        self.assertEqual(config['max_samples_per_round_jetson_orin_nx'],24)
        self.assertEqual(config['round_mode'],'BOTH')

    def test_binding_quota_uses_native_minimum_and_spacing(self):
        s=self.make_strategy(2,232)
        s.jetson_base_quota=16
        s.jetson_sample_step=8
        messages,config=self.configure(s)
        self.assertEqual({m.metadata.dst_node_id for m in messages},{1,2})
        self.assertEqual(config['max_samples_per_round_jetson_orin_nx'],24)
        self.assertEqual(s.exposure_decisions[3]['native_sample_step'],8)

    def test_below_native_minimum_omits_before_training(self):
        s=self.make_strategy(2,232)
        s.jetson_base_quota=32
        s.jetson_sample_step=8
        messages,config=self.configure(s)
        self.assertEqual([m.metadata.dst_node_id for m in messages],[1])
        self.assertEqual(config['max_samples_per_round_jetson_orin_nx'],0)

    def test_uncalibrated_warmup_narrows_only_with_labelled_proxy(self):
        s=self.make_strategy(2,232)
        s._predict_train_time=lambda name,samples,config:0.0
        _,config=self.configure(s)
        self.assertEqual(config['max_samples_per_round_jetson_orin_nx'],24)
        self.assertEqual(s.exposure_decisions[3]['candidate_cost_status'],
                         'uncalibrated_monotonic_S_proxy')
        self.assertIsNone(s.exposure_decisions[3]['predicted_residual_window_s'])

    def test_exhausted_nx_train_message_is_filtered(self):
        s=self.make_strategy(2,256)
        messages,config=self.configure(s)
        self.assertEqual([m.metadata.dst_node_id for m in messages],[1])
        self.assertEqual(config['round_mode'],'PROTECTED_ONLY')
        self.assertEqual(s.expected_train_replies_by_round[3],1)

    def test_pi_only_lora_aggregation_has_no_nx_weight(self):
        s=self.make_strategy(2,256)
        messages,_=self.configure(s)
        pi_adapter=ArrayRecord(torch_state_dict=OrderedDict({'adapter':torch.tensor([2.0])}))
        m=MetricRecord({'num-examples':4,'dataset_size':128,'partition_id':1,
            'local_steps':4,'input_tokens':256,'supervised_tokens':256,
            'train_loss':1.0,'train_time':1.0,'train_energy':1.0,
            'device_name_chars':[ord(c) for c in 'RPi5'],
            'device_uid_chars':[ord(c) for c in 'pi']})
        reply=messages[0].create_reply(RecordDict({'arrays':pi_adapter,'metrics':m}))
        arrays,_=s.aggregate_train(3,[reply])
        self.assertIsNotNone(arrays)
        self.assertTrue(torch.equal(arrays.to_torch_state_dict()['adapter'],torch.tensor([2.0])))
        self.assertEqual(s.nx_cumulative_samples,256)

    def test_partial_work_uses_actual_sample_weights_and_updates_exposure(self):
        s=self.make_strategy(2,232)
        messages,_=self.configure(s)
        replies=[]
        for message in messages:
            pi=message.metadata.dst_node_id==1
            n=4 if pi else 24
            value=1.0 if pi else 5.0
            name='RPi5' if pi else 'jetson_orin_nx'
            uid='pi' if pi else 'nx'
            record=MetricRecord({'num-examples':n,'dataset_size':128,'partition_id':1 if pi else 0,
                'local_steps':n,'input_tokens':64*n,'supervised_tokens':64*n,
                'train_loss':1.0,'train_time':1.0,'train_energy':1.0,
                'eligible_unique_samples':128,
                'visited_sample_ids':list(range(n)),
                'device_name_chars':[ord(c) for c in name],
                'device_uid_chars':[ord(c) for c in uid]})
            arrays=ArrayRecord(torch_state_dict=OrderedDict({'adapter':torch.tensor([value])}))
            replies.append(message.create_reply(RecordDict({'arrays':arrays,'metrics':record})))
        arrays,_=s.aggregate_train(3,replies)
        expected=(4*1.0+24*5.0)/28
        self.assertAlmostEqual(arrays.to_torch_state_dict()['adapter'].item(),expected,places=5)
        self.assertEqual(s.nx_cumulative_samples,256)
        self.assertEqual(s.nx_unique_visited,24)

    def test_overshoot_is_rejected_before_original_aggregation(self):
        s=self.make_strategy(2,240)
        messages,_=self.configure(s)
        nx_message=next(m for m in messages if m.metadata.dst_node_id==2)
        record=MetricRecord({'num-examples':17,'eligible_unique_samples':128,
            'visited_sample_ids':list(range(17)),
            'device_name_chars':[ord(c) for c in 'jetson_orin_nx']})
        adapter=ArrayRecord(torch_state_dict=OrderedDict({'adapter':torch.tensor([5.0])}))
        reply=nx_message.create_reply(RecordDict({'arrays':adapter,'metrics':record}))
        with patch.object(LoRACastCommAware,'aggregate_train',
                          side_effect=AssertionError('aggregation ran')):
            with self.assertRaisesRegex(RuntimeError,'scheduled exposure quota'):
                s.aggregate_train(3,[reply])
        self.assertEqual(s.nx_cumulative_samples,240)

    def test_server_app_selects_final_strategy_with_declared_policy(self):
        context=SimpleNamespace(run_config={
            'fraction-train':1.0,'min-clients':2,
            'strategy':'FinalExposureLoRACast',
            'model':'hf:Qwen/Qwen2.5-0.5B-Instruct',
            'sample-selection-mode':'exposure_aware',
            'aggregation-mode':'processed_samples',
            'nx-eligible-unique':128,'exposure-max':'2',
            'experiment-id':'isolated-wiring-test',
            'experiment-output-root':str(self.root)})
        strategy=create_strategy(context)
        self.assertIsInstance(strategy,FinalExposureLoRACast)
        self.assertEqual(strategy.exposure_max,2.0)

    def test_server_app_rejects_prefix_sampler_for_final_strategy(self):
        context=SimpleNamespace(run_config={
            'fraction-train':1.0,'min-clients':2,
            'strategy':'FinalExposureLoRACast',
            'sample-selection-mode':'prefix',
            'aggregation-mode':'processed_samples',
            'nx-eligible-unique':128,'exposure-max':'2',
            'experiment-id':'isolated-wiring-test',
            'experiment-output-root':str(self.root)})
        with self.assertRaisesRegex(ValueError,'exposure_aware sampler'):
            create_strategy(context)

    def test_round_boundary_resume_restores_exposure_state(self):
        s=self.make_strategy(2,232)
        _,_=self.configure(s)
        s.nx_seen_ids=set(range(80))
        s.nx_unique_visited=80
        s.round_stats[3]={'committed':True}
        checkpoint=self.root/'adapter_3.pt'
        torch.save({'adapter':torch.tensor([3.0])},checkpoint)
        # Test-local prediction callbacks are not part of a real strategy's
        # persistent state; remove them before exercising the production hook.
        del s.__dict__['_known_client_device_map']
        del s.__dict__['_predict_train_time']
        save_completed_round(s,3,str(checkpoint))
        restored=self.make_strategy(2,0)
        load_completed_round(restored,3,str(checkpoint))
        self.assertEqual(restored.resume_completed_round,3)
        self.assertEqual(restored.nx_cumulative_samples,232)
        self.assertEqual(restored.nx_unique_visited,80)
        self.assertEqual(restored.exposure_decisions[3]['selected_s'],24)


if __name__=='__main__':unittest.main()
