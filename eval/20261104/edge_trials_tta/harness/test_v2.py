import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import analyze as analyze
import audit_evidence as audit
import render_report as dashboard
import orchestrator as controller


class V2EvidenceTests(unittest.TestCase):
    def test_prime_can_supply_two_round_slo_without_sample_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'reference').mkdir()
            hosts={'rpi5':{'device':'RPi5','partition':1},
                   'jetson':{'device':'jetson_orin_nx','partition':0}}
            protocol={'sample_count_cap':2,'sample_offset':4,
                      'sample_split_policy':'two_disjoint_windows'}
            (root/'sample_protocol.json').write_text(json.dumps(protocol))
            prime=[]
            for host,info in hosts.items():
                observed=[]
                for round_no in (1,2,3):
                    observed.extend({'round':round_no,'device_name':info['device'],
                        'partition_id':info['partition'],'partition_train_row':i+4,
                        'token_ids_sha256':hashlib.sha256(f'{host}-official-{i}'.encode()).hexdigest(),
                        'input_token_count':128} for i in range(2))
                prime.extend({'round':round_no,'device_name':info['device'],
                    'partition_id':info['partition'],'partition_train_row':i+2,
                    'token_ids_sha256':hashlib.sha256(f'{host}-prime-{i}'.encode()).hexdigest(),
                    'input_token_count':128} for round_no in (1,2) for i in range(2))
                path=root/host/'output/sample_audit.jsonl';path.parent.mkdir(parents=True)
                path.write_text(''.join(json.dumps(row)+'\n' for row in observed))
            (root/'reference/prime_samples.jsonl').write_text(
                ''.join(json.dumps(row)+'\n' for row in prime))
            result=audit.build_dataset_audit(root,'official',(1,2,3),hosts)
            self.assertTrue(result['sample_nonoverlap_proven'])
            self.assertEqual(result['pairwise_overlap_counts'],{'prime_vs_official':0})
            self.assertEqual(result['phase_sample_offsets'],{'prime':2,'official':4})
            self.assertEqual(controller.sample_offset('prime',2,'two_disjoint_windows'),2)
            self.assertEqual(controller.sample_offset('official',2,'two_disjoint_windows'),4)
            self.assertTrue(controller.calibration_official_protocol_valid(
                protocol|{'sample_offset':2},protocol))
            self.assertFalse(controller.calibration_official_protocol_valid(
                protocol|{'sample_offset':2},protocol|{'sample_offset':2}))

    def test_prime_power_trace_has_first_round_boundary_for_slo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'server/output').mkdir(parents=True)
            base=analyze.parse_server_ts('2026-01-01_00-00-00','UTC')
            rows=[]; predictions=[]
            for number in (1,2):
                for device in ('RPi5','jetson_orin_nx'):
                    rows.append({'round':number,'phase':'train','device_name':device,
                                 'cid':device,'actual_train_time_self':1,
                                 'timestamp':f'2026-01-01_00-00-{number*10:02d}'})
                    rows.append({'round':number,'phase':'evaluate','device_name':device,
                                 'timestamp':f'2026-01-01_00-00-{number*10:02d}'})
                    predictions.append({'round':number,'cid':device,
                                        'predicted_at':base+(1 if number==1 else 11)})
            analyze.dump_csv(root/'server/output/fedavg_prime.csv',rows)
            (root/'server/output/predictions.jsonl').write_text(
                ''.join(json.dumps(item)+'\n' for item in predictions))
            for host in ('rpi5','jetson'):
                (root/host).mkdir()
                analyze.dump_csv(root/host/'power.csv',[
                    {'unix_time_s':base+i*.5,'power_w':2} for i in range(49)])
            cfg={'profiles':{'prime':{'num-server-rounds':2}},
                 'suite':{'slo_source':'prime','timezone':'UTC','max_power_sample_gap_s':3},
                 'hosts':{'rpi5':{'device':'RPi5'},'jetson':{'device':'jetson_orin_nx'}}}
            pf={'hosts':{name:{'clock':{'offset_s':0}}
                         for name in ('server','rpi5','jetson')}}
            controller.normalize(cfg,root,'prime',pf)
            normalized=analyze.read_fedavg(root/'fedavg.csv')
            self.assertEqual({float(x['prediction_recorded_at']) for x in normalized
                              if x['phase']=='train' and x['round']=='1'}, {base+1})
            obs=analyze.energy_observations(normalized,
                {device:root/host/'power_aligned.csv' for host,device in
                 (('rpi5','RPi5'),('jetson','jetson_orin_nx'))},'UTC',0,2,3)
            self.assertEqual(len(obs),4)
            self.assertAlmostEqual(obs[0]['round_energy_j'],18)

    def test_calibration_dataset_audit_creates_report_before_slo_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            hosts={'rpi5':{'device':'RPi5','partition':1},
                   'jetson':{'device':'jetson_orin_nx','partition':0}}
            (run/'state_reference').mkdir()
            (run/'sample_protocol.json').write_text(json.dumps({
                'sample_count_cap':2,'sample_offset':2,
                'sample_split_policy':'three_disjoint_windows'}))
            prime=[]
            for host,info in hosts.items():
                rows=[]
                for round_no in range(1,6):
                    rows.extend({'round':round_no,'device_name':info['device'],
                        'partition_id':info['partition'],'partition_train_row':i+2,
                        'token_ids_sha256':hashlib.sha256(f'{host}-cal-{i}'.encode()).hexdigest(),
                        'input_token_count':128} for i in range(2))
                prime.extend({'round':round_no,'device_name':info['device'],
                    'partition_id':info['partition'],'partition_train_row':i,
                    'token_ids_sha256':hashlib.sha256(f'{host}-prime-{i}'.encode()).hexdigest(),
                    'input_token_count':128} for round_no in (1,2) for i in range(2))
                path=run/host/'output/sample_audit.jsonl';path.parent.mkdir(parents=True)
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            (run/'state_reference/prime_samples.jsonl').write_text(
                ''.join(json.dumps(row)+'\n' for row in prime))
            result=audit.build_dataset_audit(run,'calibration',range(1,6),hosts)
            self.assertTrue(result['sample_nonoverlap_proven'])
            self.assertTrue((run/'report/tta_dataset_samples.csv').is_file())
            self.assertTrue((run/'report/tta_dataset_independence.json').is_file())

    def test_predispatch_and_sample_overlap_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp); (run/'server/output').mkdir(parents=True)
            (run/'reference/state_reference').mkdir(parents=True)
            cfg={'suite':{'warmup_rounds':0},'profiles':{'official':{'num-server-rounds':3}},
                 'profile_common':{'model':'gemma','dataset':'alpaca','rpi-normal-quota':2},
                 'hosts':{'rpi5':{'device':'RPi5','partition':1},'jetson':{'device':'jetson_orin_nx','partition':0}}}
            pf={'hosts':{name:{'clock':{'offset_s':0,'uncertainty_s':.05}}
                         for name in ('server','rpi5','jetson')}}
            (run/'post_clock.json').write_text(json.dumps({
                host:{'relative_drift_bound_s':.1} for host in ('rpi5','jetson')}))
            (run/'effective_config.json').write_text(json.dumps({'lora-r':16,'lora-target-modules':'q_proj,v_proj',
                                                                  'model':'gemma','dataset':'alpaca'}))
            (run/'sample_protocol.json').write_text(json.dumps({
                'sample_count_cap':2,'sample_offset':4,
                'sample_split_policy':'three_disjoint_windows'}))
            fedavg=[]; predictions=[]; prime=[]
            digest=lambda value:hashlib.sha256(value.encode()).hexdigest()
            for host,dev in [('rpi5','RPi5'),('jetson','jetson_orin_nx')]:
                (run/host/'output').mkdir(parents=True)
                (run/'reference'/host/'output').mkdir(parents=True)
                events=[]; samples=[]
                for round_no in (1,2,3):
                    start=100+10*round_no
                    fedavg.append({'round':round_no,'phase':'train','cid':host,'device_name':dev,
                                   'device_uid':host,'pred_train_time_self':1.9,
                                   'actual_train_time_self':2,'max_samples':2,
                                   'local_epochs':1,'local_steps':2,'lora_rank':16})
                    predictions.append({'round':round_no,'cid':host,'device_name':dev,
                        'predicted_at':start-5,'pred_train_time_s':1.9,
                        'history_rounds':[-1,0], 'coefficient_last_updated_round':0,
                        'config':{'batch_size':1,'max_seq_len':128},
                        'model_states':{dev:{'beta_sec_per_token':.95,'alpha':1,
                            'token_workload':2,'effective_samples':2,'effective_epochs':1,
                            'effective_steps':0,'tokens_per_example':1}}})
                    events += [{'round':round_no,'device_name':dev,'device_uid':host,
                                'unix_time_s':start,'kind':'start',
                                'workload':{'max_samples_per_round':2,'selected_samples':2,
                                            'local_steps':0,'local_epochs':1,'batch_size':1,
                                            'max_seq_len_configured':128,'sample_offset':4,
                                            'lora_rank':16,'active_lora_modules':['q_proj','v_proj'],
                                            'model':'gemma','dataset':'alpaca'}},
                               {'round':round_no,'device_name':dev,'device_uid':host,
                                'unix_time_s':start+2,'kind':'end'}]
                    samples += [{'round':round_no,'device_name':dev,'partition_id':cfg['hosts'][host]['partition'],
                                 'partition_train_row':i+4,'token_ids_sha256':digest(host+'same'+str(i)),
                                 'input_token_count':128+i} for i in (0,1)]
                prime += [{'round':r,'device_name':dev,'partition_id':cfg['hosts'][host]['partition'],
                           'partition_train_row':i,'token_ids_sha256':digest(host+'same'+str(i)),
                           'input_token_count':128+i} for r in (1,2) for i in (0,1)]
                calibration=[{'round':r,'device_name':dev,'partition_id':cfg['hosts'][host]['partition'],
                              'partition_train_row':i+2,'token_ids_sha256':digest(host+'cal'+str(i)),
                              'input_token_count':128+i} for r in (1,2,3,4,5) for i in (0,1)]
                (run/host/'output/training_events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
                (run/host/'output/sample_audit.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in samples))
                (run/'reference'/host/'output/sample_audit.jsonl').write_text(
                    ''.join(json.dumps(x)+'\n' for x in calibration))
            analyze.dump_csv(run/'fedavg.csv',fedavg)
            (run/'server/output/predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in predictions))
            (run/'reference/state_reference/prime_samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in prime))
            rows=audit.build_prediction_audit(run,cfg,pf,'official')
            self.assertEqual(len(rows),6)
            self.assertTrue(all(x['valid_prediction'] for x in rows))
            ds=json.loads((run/'report/tta_dataset_independence.json').read_text())
            self.assertEqual(ds['status'],'overlap_confirmed')
            self.assertFalse(ds['sample_nonoverlap_proven'])
            predictions[0]['history_rounds']=[-1,0,1]
            (run/'server/output/predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in predictions))
            rows=audit.build_prediction_audit(run,cfg,pf,'official')
            self.assertIn('current_or_future_round_in_history',rows[0]['invalid_reasons'])
            predictions[0]['history_rounds']=[-1,0]
            (run/'server/output/predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in predictions))
            for host in ('rpi5','jetson'):
                path=run/host/'output/sample_audit.jsonl'
                fresh=[]
                for line in path.read_text().splitlines():
                    item=json.loads(line); item['token_ids_sha256']=digest('fresh'+item['token_ids_sha256'])
                    fresh.append(item)
                path.write_text(''.join(json.dumps(x)+'\n' for x in fresh))
            rows=audit.build_prediction_audit(run,cfg,pf,'official')
            self.assertTrue(all(x['valid_prediction'] for x in rows))
            ds=json.loads((run/'report/tta_dataset_independence.json').read_text())
            self.assertEqual(ds['status'],'disjoint_observed')
            self.assertTrue(ds['sample_nonoverlap_proven'])
            self.assertEqual(ds['pairwise_overlap_counts'],{
                'prime_vs_slo_calibration':0,'prime_vs_official':0,
                'slo_calibration_vs_official':0})
            self.assertTrue((run/'report/tta_dataset_samples.csv').is_file())
            calibration_path=run/'reference/rpi5/output/sample_audit.jsonl'
            calibration=[json.loads(line) for line in calibration_path.read_text().splitlines()]
            official=[json.loads(line) for line in (run/'rpi5/output/sample_audit.jsonl').read_text().splitlines()]
            by_row={row['partition_train_row']:row['token_ids_sha256'] for row in official}
            for item in calibration:
                item['token_ids_sha256']=by_row[item['partition_train_row']+2]
            calibration_path.write_text(''.join(json.dumps(x)+'\n' for x in calibration))
            ds=audit.build_dataset_audit(run,'official',(1,2,3),cfg['hosts'])
            self.assertEqual(ds['status'],'overlap_confirmed')
            self.assertEqual(ds['pairwise_overlap_counts']['slo_calibration_vs_official'],2)
            calibration_path.unlink()
            ds=audit.build_dataset_audit(run,'official',(1,2,3),cfg['hosts'])
            self.assertEqual(ds['status'],'not_demonstrated')
            event_path=run/'rpi5/output/training_events.jsonl'
            events=[json.loads(x) for x in event_path.read_text().splitlines()]
            events[0]['workload']['batch_size']=2
            event_path.write_text(''.join(json.dumps(x)+'\n' for x in events))
            rows=audit.build_prediction_audit(run,cfg,pf,'official')
            self.assertIn('client_batch_size_mismatch',rows[0]['invalid_reasons'])

    def test_five_of_six_is_fail_even_when_one_minus_mare_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows=[]
            base=analyze.parse_server_ts('2026-01-01_00-00-00','UTC')
            for r in range(1,6):
                for dev in ('RPi5','jetson_orin_nx'):
                    pred=8.9 if r==5 and dev=='RPi5' else 10
                    rows.append({'round':r,'phase':'train','device_name':dev,
                        'timestamp':f'2026-01-01_00-00-{r*2:02d}',
                        'pred_train_time_self':pred,'actual_train_time_self':10})
                    rows.append({'round':r,'phase':'evaluate','device_name':dev,
                        'timestamp':f'2026-01-01_00-00-{r*2:02d}'})
            analyze.dump_csv(root/'fedavg.csv',rows)
            power=[]
            for dev in ('RPi5','jetson_orin_nx'):
                path=root/(dev+'.csv')
                analyze.dump_csv(path,[{'unix_time_s':base+i*.5,'power_w':2} for i in range(25)])
                power.append(dev+'='+str(path))
            (root/'slo.json').write_text(json.dumps({'slo_joules':{'RPi5':20,'jetson_orin_nx':20}}))
            args=argparse.Namespace(fedavg=str(root/'fedavg.csv'),power=power,
                 slo=str(root/'slo.json'),timezone='UTC',warmup=2,rounds=3,
                 max_gap=1,time_error_limit=.1,time_accuracy_target=.9,
                 energy_violation_target=.1,outdir=str(root/'report'))
            self.assertEqual(analyze.score(args),2)
            summary=json.loads((root/'report/tta_summary.json').read_text())
            self.assertEqual(summary['overall']['successful_predictions'],5)
            self.assertAlmostEqual(summary['overall']['time_prediction_accuracy'],5/6)
            self.assertGreater(summary['overall']['one_minus_mare_accuracy'],.9)
            self.assertFalse(summary['pass'])

    def test_pending_dashboard_never_claims_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp); (run/'state.json').write_text(json.dumps({'status':'running','stage':'official'}))
            data=dashboard.build(run)
            self.assertEqual(data['status'],'Pending')
            self.assertIn('Pending',dashboard.html(data))

    def test_mac_clock_step_does_not_invalidate_relative_alignment(self):
        before={'hosts':{'server':{'clock':{'offset_s':4.2,'uncertainty_s':.1}},
                         'rpi5':{'clock':{'offset_s':4.25,'uncertainty_s':.1}}}}
        after={'server':{'offset_s':.2,'uncertainty_s':.1},
               'rpi5':{'offset_s':.25,'uncertainty_s':.1}}
        checked=controller.audit_relative_clock(before,after,['rpi5'],1.0)
        self.assertLess(checked['rpi5']['relative_drift_bound_s'],1.0)
        after['rpi5']={'offset_s':1.5,'uncertainty_s':.1}
        with self.assertRaises(controller.Failure):
            controller.audit_relative_clock(before,after,['rpi5'],1.0)

    def test_slo_audit_rejects_posthoc_or_changed_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ref=root/'calibration'; run=root/'official'
            (run/'reference').mkdir(parents=True);(run/'report').mkdir()
            ref.mkdir()
            (ref/'state.json').write_text(json.dumps({'stage':'calibration','status':'validated'}))
            protocol={'sample_count_cap':2,'sample_offset':2,
                      'sample_split_policy':'three_disjoint_windows'}
            (run/'sample_protocol.json').write_text(json.dumps(protocol|{'sample_offset':4}))
            (ref/'sample_protocol.json').write_text(json.dumps(protocol))
            (ref/'fedavg.csv').write_text('raw calibration\n')
            (ref/'suite.toml').write_text('fixed config\n')
            for host in ('rpi5','jetson'):
                (ref/host).mkdir();(ref/host/'power_aligned.csv').write_text(host+' power\n')
            energy={'RPi5':[10,11,12,13,14],'jetson_orin_nx':[20,21,22,23,24]}
            created='2026-10-10T09:00:00+09:00'
            lock={'created_at':created,'margin':1.10,'measurement_rounds':5,
                  'calibration_round_numbers':[1,2,3,4,5],
                  'calibration_energy_j':energy,
                  'slo_joules':{d:max(v)*1.10 for d,v in energy.items()},
                  'formula':'device_slo_j = max(calibration_whole_round_energy_j) * margin',
                  'sample_protocol':protocol,'workload_config':{'max-samples-per-round':2},
                  'calibration_raw_csv_sha256':controller.digest(ref/'fedavg.csv'),
                  'tta_suite_sha256':controller.digest(ref/'suite.toml'),
                  'power_trace_sha256':{d:controller.digest(ref/host/'power_aligned.csv')
                                        for d,host in [('RPi5','rpi5'),('jetson_orin_nx','jetson')]}}
            (ref/'slo_lock.json').write_text(json.dumps(lock))
            (run/'reference/slo_lock.json').write_text(json.dumps(lock))
            start=controller.digest(ref/'slo_lock.json')
            (run/'slo_start.json').write_text(json.dumps({'slo_lock_sha256':start,
                'recorded_at':datetime.fromisoformat(created).timestamp()+100}))
            result=controller.slo_audit(run,ref,start)
            self.assertTrue(result['unchanged'])
            self.assertTrue(result['lock_preceded_official'])
            self.assertTrue(result['calibration_protocol_valid'])
            self.assertFalse(controller.calibration_official_protocol_valid(
                protocol,protocol))
            self.assertFalse(controller.calibration_official_protocol_valid(
                protocol,protocol|{'sample_offset':4,'max_seq_len':256}))
            (run/'slo_start.json').write_text(json.dumps({'slo_lock_sha256':start,
                'recorded_at':datetime.fromisoformat(created).timestamp()-100}))
            with self.assertRaises(controller.Failure):
                controller.slo_audit(run,ref,start)
            (run/'slo_start.json').write_text(json.dumps({'slo_lock_sha256':start,
                'recorded_at':datetime.fromisoformat(created).timestamp()+100}))
            lock['margin']=1.20
            (run/'reference/slo_lock.json').write_text(json.dumps(lock))
            with self.assertRaises(controller.Failure):
                controller.slo_audit(run,ref,start)

    def test_merged_prime_slo_audit_requires_equal_sample_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref=root/'prime';run=root/'official'
            (run/'reference').mkdir(parents=True);(run/'report').mkdir()
            ref.mkdir()
            (ref/'state.json').write_text(json.dumps({'stage':'prime','status':'validated'}))
            protocol={'sample_count_cap':2,'sample_offset':2,
                      'sample_split_policy':'two_disjoint_windows'}
            (ref/'sample_protocol.json').write_text(json.dumps(protocol))
            (run/'sample_protocol.json').write_text(json.dumps(protocol|{'sample_offset':4}))
            (ref/'fedavg.csv').write_text('prime observations\n')
            (ref/'suite.toml').write_text('fixed config\n')
            for host in ('rpi5','jetson'):
                (ref/host).mkdir()
                (ref/host/'power_aligned.csv').write_text(host+' trace\n')
            rows=[{'phase':phase,'device_name':dev,'input_token_count':128}
                  for phase in ('prime','official')
                  for dev in ('RPi5','jetson_orin_nx') for _ in range(2)]
            analyze.dump_csv(run/'report/tta_dataset_samples.csv',rows)
            created='2026-10-10T09:00:00+09:00'
            energy={'RPi5':[10,11],'jetson_orin_nx':[20,21]}
            lock={'created_at':created,'margin':1.10,'measurement_rounds':2,
                  'calibration_round_numbers':[1,2],'slo_source':'prime',
                  'calibration_energy_j':energy,
                  'slo_joules':{d:max(v)*1.10 for d,v in energy.items()},
                  'formula':'device_slo_j = max(calibration_whole_round_energy_j) * margin',
                  'sample_protocol':protocol,'workload_config':{'max-samples-per-round':2},
                  'calibration_raw_csv_sha256':controller.digest(ref/'fedavg.csv'),
                  'tta_suite_sha256':controller.digest(ref/'suite.toml'),
                  'power_trace_sha256':{d:controller.digest(ref/host/'power_aligned.csv')
                                        for d,host in [('RPi5','rpi5'),('jetson_orin_nx','jetson')]}}
            (ref/'slo_lock.json').write_text(json.dumps(lock))
            (run/'reference/slo_lock.json').write_text(json.dumps(lock))
            start=controller.digest(ref/'slo_lock.json')
            (run/'slo_start.json').write_text(json.dumps({'slo_lock_sha256':start,
                'recorded_at':datetime.fromisoformat(created).timestamp()+100}))
            result=controller.slo_audit(run,ref,start)
            self.assertTrue(result['calibration_protocol_valid'])
            self.assertTrue(result['calibration_sample_lengths_match_official'])
            rows[-1]['input_token_count']=127
            analyze.dump_csv(run/'report/tta_dataset_samples.csv',rows)
            with self.assertRaisesRegex(controller.Failure,'calibration evidence mismatch'):
                controller.slo_audit(run,ref,start)


if __name__=='__main__': unittest.main()
