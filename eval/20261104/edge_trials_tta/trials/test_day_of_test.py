import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
from io import StringIO
import json
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import day_of_test as day


class DayWorkflowTests(unittest.TestCase):
    def test_force_preserves_old_selection_until_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old=self.calibration(root)
            new=root/'new_calibration';shutil.copytree(old,new)
            config=root/'config.toml';config.write_text('[suite]\n')
            fixed=datetime(2026,10,10,9,16,tzinfo=ZoneInfo('Asia/Seoul'))
            args=argparse.Namespace(config=config,state=None,force=True)
            with patch.object(day,'SESSIONS',root/'sessions'), \
                 patch.object(day,'seoul_now',return_value=fixed), \
                 patch.object(day,'current_fingerprint',return_value='fixed-config'), \
                 patch.object(day,'prepare_wifi_route'), redirect_stdout(StringIO()):
                day.select_calibration(old,config)
                selected=day.selected_path(config=config);original=selected.read_bytes()
                with patch.object(day,'stream',return_value=(2,None)):
                    self.assertEqual(day.prepare(args),2)
                self.assertEqual(selected.read_bytes(),original)
                with patch.object(day,'stream',side_effect=[(0,root/'prime'),(0,new)]):
                    self.assertEqual(day.prepare(args),0)
                record,_=day.load_selected(config=config)
                self.assertEqual(record['calibration_run'],str(new.resolve()))
                self.assertEqual(Path(record['supersedes_selection']).read_bytes(),original)
                self.assertTrue((old/'slo_lock.json').exists())

    def test_model_dataset_override_is_immutable_and_isolated(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            base=root/'suite.toml'
            original=('[suite]\nslo_source = "prime"\n'
                      '[profile_common]\nmodel = "hf:base"\n'
                      'dataset = "data/base"\nmax-seq-len = 128\n'
                      '[profiles.official]\nnum-server-rounds = 3\n')
            base.write_text(original)
            a=day.materialize_config(base,'hf:google/gemma-3-270m',
                                     'yahma/alpaca-cleaned',root/'generated')
            self.assertEqual(base.read_text(),original)
            self.assertEqual(a,day.materialize_config(base,'hf:google/gemma-3-270m',
                                     'yahma/alpaca-cleaned',root/'generated'))
            config=tomllib.loads(a.read_text())
            self.assertEqual(config['profile_common']['model'],'hf:google/gemma-3-270m')
            self.assertEqual(config['profile_common']['dataset'],'yahma/alpaca-cleaned')
            self.assertEqual(config['profile_common']['max-seq-len'],128)
            other=day.materialize_config(base,'hf:other',None,root/'generated')
            self.assertNotEqual(a,other)
            self.assertEqual(tomllib.loads(other.read_text())['profile_common']['dataset'],'data/base')
            with self.assertRaisesRegex(ValueError,'한 줄'):
                day.materialize_config(base,'hf:bad\nmodel',None,root/'generated')

    def calibration(self, root):
        run = root / 'calibration'
        run.mkdir()
        (run / 'state.json').write_text(json.dumps({'status':'validated','stage':'calibration'}))
        lock = {'created_at':'2026-10-10T09:15:00+09:00',
                'warmup_rounds':0,'measurement_rounds':5,'margin':1.1,
                'slo_joules':{'RPi5':412.5,'jetson_orin_nx':655.0},
                'test_fingerprint':'fixed-config','state_sha256':'fixed-prime-state'}
        (run / 'slo_lock.json').write_text(json.dumps(lock))
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in run.iterdir() if p.is_file()}
        (run / 'checksums.json').write_text(json.dumps(hashes))
        return run

    def test_prepare_selects_once_and_onsite_uses_exact_reference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            run=self.calibration(root)
            config=root/'config.toml';config.write_text('[suite]\n')
            prime=root/'prime';prime.mkdir()
            args=argparse.Namespace(config=config,state=prime,no_open=True)
            fixed=datetime(2026,10,10,9,16,tzinfo=ZoneInfo('Asia/Seoul'))
            with patch.object(day,'SESSIONS',root/'sessions'), \
                 patch.object(day,'seoul_now',return_value=fixed), \
                 patch.object(day,'prepare_wifi_route'), \
                 patch.object(day,'current_fingerprint',return_value='fixed-config'), \
                 patch.object(day,'stream',return_value=(0,run)) as stream:
                self.assertEqual(day.prepare(args),0)
                record,lock=day.load_selected()
                self.assertEqual(record['calibration_run'],str(run.resolve()))
                self.assertEqual(record['slo_lock_sha256'],day.sha(run/'slo_lock.json'))
                output=StringIO()
                with redirect_stdout(output):
                    self.assertEqual(day.prepare(args),0)
                self.assertIn('오늘 성공한 보정이 이미 고정',output.getvalue())
                for name in ('slo_lock.json','fedavg.csv','rpi5/power_aligned.csv',
                             'jetson/power_aligned.csv','rpi5/power.csv',
                             'jetson/power.csv','checksums.sha256'):
                    self.assertIn(str(run.resolve()/name),output.getvalue())
                stream.assert_called_once()
                with patch.object(day.subprocess,'run',return_value=argparse.Namespace(returncode=0)) as launch:
                    self.assertEqual(day.onsite(args),0)
                    argv=launch.call_args.args[0]
                    self.assertEqual(argv[argv.index('--reference')+1],str(run.resolve()))
                    self.assertEqual(argv[argv.index('--config')+1],str(config.resolve()))
                    self.assertIn('--no-open',argv)
                (run/'slo_lock.json').write_text('{}')
                with self.assertRaises(ValueError):day.load_selected()

    def test_prime_mode_selects_two_round_slo_without_second_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=self.calibration(root)
            state={'status':'validated','stage':'prime'}
            (run/'state.json').write_text(json.dumps(state))
            lock=json.loads((run/'slo_lock.json').read_text())
            lock.update(slo_source='prime',measurement_rounds=2)
            (run/'slo_lock.json').write_text(json.dumps(lock))
            hashes={p.name:day.sha(p) for p in run.iterdir()
                    if p.is_file() and p.name!='checksums.json'}
            (run/'checksums.json').write_text(json.dumps(hashes))
            config=root/'config.toml';config.write_text('[suite]\nslo_source="prime"\n')
            args=argparse.Namespace(config=config,state=None,no_open=True)
            fixed=datetime(2026,10,10,9,16,tzinfo=ZoneInfo('Asia/Seoul'))
            with patch.object(day,'SESSIONS',root/'sessions'), \
                 patch.object(day,'seoul_now',return_value=fixed), \
                 patch.object(day,'prepare_wifi_route'), \
                 patch.object(day,'current_fingerprint',return_value='fixed-config'), \
                 patch.object(day,'stream',return_value=(0,run)) as stream:
                self.assertEqual(day.prepare(args),0)
                self.assertEqual(stream.call_count,1)
                self.assertEqual(stream.call_args.args[0][-1],'prime')
                selected,selected_lock=day.load_selected()
                self.assertEqual(selected['calibration_run'],str(run.resolve()))
                self.assertEqual(selected_lock['measurement_rounds'],2)

    def test_no_successful_calibration_is_not_selected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            config=root/'config.toml';config.write_text('[suite]\n')
            prime=root/'prime';prime.mkdir()
            args=argparse.Namespace(config=config,state=prime,no_open=True)
            fixed=datetime(2026,10,10,9,16,tzinfo=ZoneInfo('Asia/Seoul'))
            with patch.object(day,'SESSIONS',root/'sessions'), \
                 patch.object(day,'seoul_now',return_value=fixed), \
                 patch.object(day,'prepare_wifi_route'), \
                 patch.object(day,'current_fingerprint',return_value='fixed-config'), \
                 patch.object(day,'stream',return_value=(2,root/'failed')):
                self.assertEqual(day.prepare(args),2)
                self.assertFalse(day.selected_path().exists())

    def test_new_protocol_keeps_old_same_day_record_and_cannot_replace_its_slo(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); run=self.calibration(root)
            lock=json.loads((run/'slo_lock.json').read_text())
            lock['test_fingerprint']='new-config'
            (run/'slo_lock.json').write_text(json.dumps(lock))
            hashes=json.loads((run/'checksums.json').read_text())
            hashes['slo_lock.json']=day.sha(run/'slo_lock.json')
            (run/'checksums.json').write_text(json.dumps(hashes))
            second=root/'second';shutil.copytree(run,second)
            config=root/'config.toml';config.write_text('[suite]\n')
            sessions=root/'sessions';sessions.mkdir()
            legacy=sessions/'2026-10-10.json'
            legacy.write_text(json.dumps({'test_fingerprint':'old-config'}))
            fixed=datetime(2026,10,10,9,16,tzinfo=ZoneInfo('Asia/Seoul'))
            with patch.object(day,'SESSIONS',sessions), \
                 patch.object(day,'seoul_now',return_value=fixed), \
                 patch.object(day,'current_fingerprint',return_value='new-config'):
                self.assertEqual(day.select_calibration(run,config),0)
                self.assertEqual(json.loads(legacy.read_text())['test_fingerprint'],'old-config')
                self.assertTrue(day.selected_path().exists())
                with self.assertRaisesRegex(ValueError,'이미 선택'):
                    day.select_calibration(second,config)


if __name__=='__main__':unittest.main()
