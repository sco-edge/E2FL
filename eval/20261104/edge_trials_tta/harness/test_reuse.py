import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import analyze as analyze
import audit_evidence
import orchestrator as orchestrator
import predictor_state


class ReuseTests(unittest.TestCase):
    def test_historical_prime_needs_no_future_sample_manifest(self):
        cfg={'suite':{'token_input_mode':'historical'},'profile_common':{},
             'server':{},'hosts':{}}
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)
            for host in ('rpi5','jetson'):
                output=run/host/'output'; output.mkdir(parents=True)
                (output/'sample_audit.jsonl').write_text(
                    json.dumps({'round':1,'device_name':host,'token_ids_sha256':host})+'\n')
            orchestrator.record_prime_samples(run)
            self.assertTrue((run/'prime_samples.jsonl').is_file())
            self.assertFalse((run/'planned_samples.jsonl').exists())
            predictor_state.write(run/'predictor_state.json',self.state())
            orchestrator.save(run/'predictor_manifest.json',{
                'test_fingerprint':orchestrator.fingerprint(cfg),
                'state_sha256':orchestrator.digest(run/'predictor_state.json')})
            orchestrator.save(run/'checksums.json',{})
            self.assertEqual(orchestrator.verify_prime(cfg,run),run.resolve())

    def test_default_mode_does_not_install_future_token_inputs(self):
        cfg={'suite':{'warmup_rounds':0,'token_input_mode':'historical'},
             'profile_common':{'rpi-normal-quota':2,'max-seq-len':128},
             'server':{'control_address':'127.0.0.1:19093'}}
        with tempfile.TemporaryDirectory() as d:
            app=Path(d)
            (app/'pyproject.toml').write_text('[tool.flwr.app.config]\n"batch_size" = 1\n')
            orchestrator.prepare_app(cfg,app,{},'/tmp/tta','prime')
            settings=(app/'run_settings.py').read_text()
            self.assertIn('PLANNED_SAMPLE_COUNT = 0',settings)
            self.assertIn('PLANNED_SAMPLE_OFFSET = 4',settings)
            self.assertIn('PLANNED_TOKENS_PER_EXAMPLE = {}',settings)
            self.assertIn('TTA_MAX_SEQ_LEN = 128',settings)
            cfg['suite']['token_input_mode']='predeclared_diagnostic'
            orchestrator.prepare_app(cfg,app,{},'/tmp/tta','prime')
            self.assertIn('PLANNED_SAMPLE_COUNT = 2',
                          (app/'run_settings.py').read_text())

    def test_sequence_cap_patch_and_actual_sample_audit(self):
        source=('from e2fl.task import is_hf_model_name\n'
                'a = tokenizer(text, truncation=True, padding=False)\n'
                'b = tokenizer(texts, truncation=True, padding=False)\n')
        patched=orchestrator.enforce_sequence_cap(source)
        self.assertEqual(patched.count('max_length=TTA_MAX_SEQ_LEN'),2)
        self.assertIn('from run_settings import TTA_MAX_SEQ_LEN',patched)
        cfg={'profile_common':{'max-seq-len':128},
             'profiles':{'prime':{'num-server-rounds':2}},
             'hosts':{'rpi5':{'device':'RPi5'},
                      'jetson':{'device':'jetson_orin_nx'}}}
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)
            (run/'sample_protocol.json').write_text(json.dumps({
                'sequence_length_cap_enforced':True,'max_seq_len':128}))
            fedavg=[]
            for host,info in cfg['hosts'].items():
                output=run/host/'output';output.mkdir(parents=True)
                samples=[]
                for round_number in (1,2):
                    fedavg.append({'round':round_number,'phase':'train',
                        'device_name':info['device'],'num_examples':2,
                        'local_epochs':1,'input_tokens':255})
                    samples.extend({'round':round_number,'input_token_count':length}
                                   for length in (127,128))
                (output/'sample_audit.jsonl').write_text(''.join(
                    json.dumps(row)+'\n' for row in samples))
            analyze.dump_csv(run/'fedavg.csv',fedavg)
            self.assertEqual(audit_evidence.verify_sequence_cap(run,cfg,'prime')[(1,'RPi5')],128)
            path=run/'rpi5/output/sample_audit.jsonl'
            samples=[json.loads(line) for line in path.read_text().splitlines()]
            samples[0]['input_token_count']=129
            path.write_text(''.join(json.dumps(row)+'\n' for row in samples))
            with self.assertRaisesRegex(ValueError,'exceeds 128-token cap'):
                audit_evidence.verify_sequence_cap(run,cfg,'prime')

    def test_planned_sample_features_are_frozen_before_training(self):
        cfg={'profile_common':{'rpi-normal-quota':2,'model':'gemma','dataset':'alpaca'},
             'hosts':{'rpi5':{'device':'RPi5','partition':1},
                      'jetson':{'device':'jetson_orin_nx','partition':0}}}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); prime=root/'prime'; prime.mkdir()
            planned=[]; prime_samples=[]; fedavg=[]
            for host,info in cfg['hosts'].items():
                dev=info['device']; out=root/host/'output'; out.mkdir(parents=True)
                observed=[]
                for index,length in ((4,120),(5,210)):
                    feature={'device_name':dev,'device_uid':host,'partition_id':info['partition'],
                             'partition_train_row':index,'token_ids_sha256':f'{info["partition"]}{index}'.ljust(64,'a'),
                             'input_token_count':length}
                    planned.append(feature | {'model':'gemma','dataset':'alpaca','recorded_at':10})
                    for round_number in (1,2,3):
                        observed.append(feature | {'round':round_number})
                for index in (0,1):
                    prime_samples.append({'token_ids_sha256':f'prime-{dev}-{index}'})
                (out/'sample_audit.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in observed))
                for round_number in (1,2,3):
                    fedavg.append({'round':round_number,'phase':'train','device_name':dev,
                                   'local_epochs':1,'input_tokens':330})
            (prime/'planned_samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in planned))
            (prime/'prime_samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in prime_samples))
            manifest=audit_evidence.build_planned_feature_manifest(prime,cfg)
            self.assertEqual(manifest['devices']['RPi5']['input_tokens_per_epoch'],330)
            duplicate=planned[0] | {'recorded_at':20}
            with (prime/'planned_samples.jsonl').open('a') as stream:
                stream.write(json.dumps(duplicate)+'\n')
            self.assertEqual(audit_evidence.build_planned_feature_manifest(prime,cfg)['last_recorded_at'],10)
            with (prime/'planned_samples.jsonl').open('a') as stream:
                stream.write(json.dumps(duplicate | {'input_token_count':121})+'\n')
            with self.assertRaises(ValueError):
                audit_evidence.build_planned_feature_manifest(prime,cfg)
            analyze.dump_csv(root/'fedavg.csv',fedavg)
            self.assertFalse(any(audit_evidence.planned_sample_mismatches(root,cfg,manifest,(1,2,3)).values()))
            fedavg[0]['input_tokens']=329
            analyze.dump_csv(root/'fedavg.csv',fedavg)
            self.assertIn('planned_token_workload_mismatch',
                          audit_evidence.planned_sample_mismatches(root,cfg,manifest,(1,))[1,'RPi5'])

    def test_wifi_health_invalidates_firmware_restart_or_route_fallback(self):
        before={'boot_id':'boot-a','firmware_crash_count':5,
                'route':{'dev':'wlP1p1s0','prefsrc':'192.168.128.88'},
                'wifi_link':{'ssid':'NetLab_5G','bssid':'b0:38:6c:54:65:84','frequency_mhz':5180},
                'wifi_power_save':'Power save: off','wifi_11ax_disabled':'Y'}
        orchestrator.validate_jetson_wifi_health(before,before.copy())
        for changed in ({'firmware_crash_count':6},
                        {'route':{'dev':'enP8p1s0','prefsrc':'192.168.128.89'}},
                        {'wifi_power_save':'Power save: on'},
                        {'wifi_link':{'ssid':'NetLab_2.4G','bssid':'b0:38:6c:54:65:83','frequency_mhz':2412}},
                        {'wifi_11ax_disabled':'N'},
                        {'boot_id':'boot-b'}):
            with self.subTest(changed=changed):
                with self.assertRaises(orchestrator.Failure):
                    orchestrator.validate_jetson_wifi_health(before,before | changed)

    def test_wifi_link_records_actual_ap_and_band(self):
        link=orchestrator.parse_wifi_link('Connected to B0:38:6C:54:65:84 (on wlP1p1s0)\n\tSSID: NetLab_5G\n\tfreq: 5180\n')
        self.assertEqual(link,{'ssid':'NetLab_5G','bssid':'b0:38:6c:54:65:84','frequency_mhz':5180})
        self.assertEqual(orchestrator.parse_wifi_link('Connected to b0:38:6c:54:65:83\nSSID: NetLab_2.4G\nfreq: 2412.0\n')['frequency_mhz'],2412)
        with self.assertRaises(orchestrator.Failure):
            orchestrator.parse_wifi_link('Not connected.\n')

    def state(self):
        entry = lambda seconds: {'tokens': 128, 'train_time': seconds}
        return {'schema': 1, 'source_rounds': [1, 2],
                'round_stats': {'1': {'RPi5': entry(20), 'jetson_orin_nx': entry(2)},
                                '2': {'RPi5': entry(22), 'jetson_orin_nx': entry(3)}},
                'window_history': [{'RPi5': 128, 'jetson_orin_nx': 128}] * 2,
                'slowest_device_name': 'RPi5',
                **{name: {} for name in predictor_state.TABLES}}

    def test_restore_uses_history_without_reusing_node_ids(self):
        state = self.state()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'state.json'
            predictor_state.write(path, state)
            fresh = SimpleNamespace(round_stats={}, window_history=[], min_beta_round=1,
                                    slowest_device_name=None, cid_to_device={77: 'RPi5'},
                                    cid_to_device_uid={77: 'rpi'},
                                    **{name: {} for name in predictor_state.TABLES})
            predictor_state.restore(fresh, predictor_state.read(path))
            self.assertEqual(set(fresh.round_stats), {-1, 0})
            self.assertEqual(fresh.min_beta_round, -1)
            self.assertEqual(fresh.cid_to_device, {77: 'RPi5'})
            self.assertEqual(fresh.cid_to_device_uid, {77: 'rpi'})
            with self.assertRaises(ValueError):
                predictor_state.restore(fresh, state)

    def test_rejects_incomplete_or_nonfinite_state(self):
        state = self.state()
        del state['round_stats']['2']['RPi5']
        with self.assertRaises(ValueError): predictor_state.validate(state)
        state = self.state()
        state['time_scale']['RPi5'] = float('nan')
        with self.assertRaises(ValueError): predictor_state.validate(state)

    def test_first_live_round_uses_predispatch_energy_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            start = analyze.parse_server_ts('2026-01-01_00-00-00', 'UTC')
            rows=[]
            for round_no in (1, 2, 3):
                for dev in predictor_state.GROUPS:
                    rows += [dict(round=round_no, phase='train', device_name=dev,
                                  timestamp=f'2026-01-01_00-00-{round_no*2:02d}',
                                  prediction_recorded_at=str(start + .5) if round_no==1 else '',
                                  pred_train_time_self='1.9', actual_train_time_self='2'),
                             dict(round=round_no, phase='evaluate', device_name=dev,
                                  timestamp=f'2026-01-01_00-00-{round_no*2:02d}')]
            analyze.dump_csv(root/'fedavg.csv', rows)
            power={}
            for dev in predictor_state.GROUPS:
                path=root/(dev+'.csv')
                analyze.dump_csv(path, [dict(unix_time_s=start+i*.5,power_w=3) for i in range(15)])
                power[dev]=path
            observed=analyze.energy_observations(rows,power,'UTC',0,3,1)
            self.assertEqual(len(observed),6)
            self.assertAlmostEqual(observed[0]['round_energy_j'],4.5)
            args=argparse.Namespace(fedavg=str(root/'fedavg.csv'),
                                    power=[f'{d}={p}' for d,p in power.items()],
                                    timezone='UTC',warmup=0,rounds=3,margin=1.1,max_gap=1,
                                    out=str(root/'slo.json'),slo=str(root/'slo.json'),
                                    outdir=str(root/'report'),time_error_limit=.1,
                                    time_accuracy_target=.9,energy_violation_target=.1,
                                    accuracy_metric='within_error_limit')
            analyze.lock_slo(args)
            self.assertEqual(analyze.score(args),2)
            summary=json.loads((root/'report/tta_summary.json').read_text())
            self.assertIn('pre-dispatch',summary['criteria']['energy_definition'])


if __name__=='__main__': unittest.main()
