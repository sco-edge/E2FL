import csv
import json
import pickle
from argparse import Namespace
from pathlib import Path
import tempfile
import tomllib
import unittest

import policy_compare as p
import policy_compare_report as comparison_report
import policy_prediction_audit


class PolicyComparisonTests(unittest.TestCase):
    def test_predispatch_audit_does_not_mutate_predictor(self):
        class Fake:
            def configure_train(self,*args):return []
            def _predict_train_time(self,key,samples,cfg):
                self.cache[key]=samples
                return samples*2
            def _estimate_tokens_per_example(self,*args):return 64
        with tempfile.TemporaryDirectory() as folder:
            strategy=Fake();strategy.log_dir=folder
            strategy.round_configs={3:{'max_samples_per_round':9}}
            strategy.round_stats={1:{},2:{}}
            strategy.cid_to_device={};strategy.cid_to_device_uid={};strategy.cache={}
            policy_prediction_audit.install(strategy)
            self.assertEqual(strategy.configure_train(3,None,{},None),[])
            self.assertEqual(strategy.cache,{})
            self.assertNotIn('configure_train',strategy.__dict__)
            pickle.dumps(strategy.__dict__)
            records=[json.loads(s) for s in (Path(folder)/'policy_predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),2)
            self.assertEqual(records[0]['predicted_s'],18)
            self.assertEqual(records[0]['history_rounds'],[1,2])

    def args(self, mode, cap=None, exposure=None):
        return Namespace(mode=mode,model='hf:google/gemma-3-270m',
            dataset='yahma/alpaca-cleaned',rounds=5,pi_quota=2,
            jetson_max=16,max_seq_len=128,seed=47,
            tbm_cap_samples=cap,exposure_max=exposure)

    def test_three_modes_share_model_dataset_sampler_and_seed(self):
        manifest=json.loads(p.DEFAULT_MANIFEST.read_text())
        configs=[p.profile(self.args('uniform'),manifest),
                 p.profile(self.args('loracast-off'),manifest),
                 p.profile(self.args('loracast-on',10),manifest)]
        for key in ('model','dataset','sample-selection-mode','sample-seed',
                    'partition-manifest-path','rpi-normal-quota','max-seq-len'):
            self.assertEqual(len({str(x.get(key)) for x in configs}),1,key)
        self.assertEqual([x['jetson-max-quota'] for x in configs],[2,16,16])
        self.assertEqual([x['exposure-max'] for x in configs[:2]],['none','none'])
        self.assertEqual(int(float(configs[2]['exposure-max'])*23292),10)
        paper=p.profile(self.args('loracast-on',exposure=2),manifest)
        self.assertEqual(paper['exposure-max'],'2')
        with self.assertRaisesRegex(ValueError,'requires'):
            p.profile(self.args('loracast-on'),manifest)
        with self.assertRaisesRegex(ValueError,'exactly one'):
            p.profile(self.args('loracast-on',cap=10,exposure=2),manifest)
        with self.assertRaisesRegex(ValueError,'manifest dataset'):
            p.profile(self.args('loracast-off',None),{**manifest,'dataset_name':'other'})

    def test_staged_flower_config_uses_isolated_local_deployment(self):
        manifest=json.loads(p.DEFAULT_MANIFEST.read_text())
        config=p.profile(self.args('loracast-on',10),manifest)
        rendered=p.render_pyproject(p.SOURCE_APP/'pyproject.toml',config)
        parsed=tomllib.loads(rendered)
        self.assertEqual(parsed['tool']['flwr']['app']['config']['model'],
                         'hf:google/gemma-3-270m')
        self.assertEqual(parsed['tool']['flwr']['federations']['local-deployment']['address'],
                         '127.0.0.1:19093')
        self.assertEqual(parsed['tool']['flwr']['app']['config']['exposure-max'],
                         config['exposure-max'])

    def test_report_requires_explicit_omission(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            cond={'config':{'experiment-id':'trial'},'mode':'loracast-on',
                  'model':'gemma','dataset':'alpaca','manifest_sha256':'sha',
                  'rounds':3,'tbm_cap_samples':2}
            p.core.save(run/'condition.json',cond)
            server=run/'server/output/trial/server';server.mkdir(parents=True)
            fields=['round','phase','device_name','device_uid','num_examples',
                    'input_tokens','local_steps','actual_train_time_self',
                    'actual_upload_time_self','round_mode']
            rows=[]
            for r in (1,2,3):
                for device in (('RPi5','jetson_orin_nx') if r<3 else ('RPi5',)):
                    rows.append(dict(round=r,phase='train',device_name=device,
                        device_uid=device,num_examples=2,input_tokens=256,
                        local_steps=2,actual_train_time_self=10,
                        actual_upload_time_self=1,round_mode='BOTH'))
            with (server/'fedavg_trial.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            with (server/'aggregation.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['round','device_uid','aggregation_weight'])
                writer.writeheader()
                for row in rows:
                    writer.writerow({'round':row['round'],'device_uid':row['device_uid'],
                                     'aggregation_weight':1 if row['round']==3 else .5})
            with self.assertRaisesRegex(ValueError,'without an explicit'):
                p.summarize(run)
            p.core.save(server/'final_exposure_decision_round_003.json',
                        {'action':'omit_before_training','remaining_before':0})
            report=p.summarize(run)
            lines=p.read_rows(report)
            self.assertEqual(len(lines),6)
            self.assertEqual(lines[-1]['tbm_action'],'omit_before_training')
            self.assertEqual(p.core.read(run/'policy_summary.json')['nx_omitted_rounds'],[3])
            self.assertIsNone(p.core.read(run/'policy_summary.json')['tta_pass'])

    def test_logged_prediction_check_keeps_audit_unverified(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            server=run/'server/output/trial/server';server.mkdir(parents=True)
            path=server/'fedavg_trial.csv'
            condition={'mode':'loracast-off','warmup_rounds':2,'rounds':5,
                       'config':{'experiment-id':'trial'}}
            data=[{'phase':'train','round':r,'device_name':device,
                   'pred_train_time_self':11,'actual_train_time_self':10}
                  for r in range(3,6) for device in ('RPi5','jetson_orin_nx')]
            def write_data():
                with path.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(data[0]))
                    writer.writeheader();writer.writerows(data)
            write_data()
            summary,details=comparison_report.prediction_check(run,condition)
            self.assertEqual(summary['successful_predictions'],6)
            self.assertEqual(summary['numerical_status'],'meets_target')
            self.assertEqual(summary['prediction_audit_status'],'not_verified')
            self.assertIsNone(summary['tta_pass'])
            data[0]['pred_train_time_self']=12
            write_data()
            summary,_=comparison_report.prediction_check(run,condition)
            self.assertEqual(summary['successful_predictions'],5)
            self.assertEqual(summary['numerical_status'],'below_target')
            data[0]['pred_train_time_self']='nan'
            write_data()
            summary,details=comparison_report.prediction_check(run,condition)
            self.assertEqual(summary['numerical_status'],'incomplete')
            self.assertEqual(summary['observations'],5)
            self.assertEqual(details[0]['observation_status'],'not_measured')
            audit=server/'policy_predictions.jsonl'
            audit.write_text(json.dumps({'round':3,'device_name':'RPi5',
                'predicted_s':10,'history_rounds':[1,2],'predicted_at':100})+'\n')
            summary,details=comparison_report.prediction_check(run,condition)
            self.assertEqual(summary['prediction_source'],'before_dispatch')
            self.assertEqual(summary['observations'],1)
            self.assertEqual(summary['successful_predictions'],1)
            self.assertEqual(summary['numerical_status'],'incomplete')
            audit.write_text(json.dumps({'round':3,'device_name':'RPi5',
                'predicted_s':10,'history_rounds':[1,2,3]})+'\n')
            with self.assertRaisesRegex(ValueError,'current/future'):
                comparison_report.prediction_check(run,condition)


if __name__=='__main__':unittest.main()
