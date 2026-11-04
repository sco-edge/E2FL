import argparse, csv, importlib.util, json, math, os, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from unittest.mock import patch
import analyze as a
import orchestrator as c

class AnalysisTests(unittest.TestCase):
    def test_trapezoid(self):
        self.assertEqual(a.integrate([(0,2),(1,4),(2,6)],.5,1.5,2)[0],4)
    def test_gap_across_boundary(self):
        with self.assertRaises(ValueError): a.integrate([(0,2),(10,2)],4,5,3)
    def test_no_extrapolation(self):
        with self.assertRaises(ValueError): a.integrate([(0,2),(1,2)],.5,2,3)
    def test_missing_round(self):
        with self.assertRaises(ValueError): a.measured_rounds([{'phase':'train','round':3},{'phase':'train','round':5}],2,2)
    def fixture(self,root,duplicate=False,nan=False):
        from datetime import datetime
        rows=[]
        for r in range(1,6):
            for dev in ['RPi5','jetson_orin_nx']:
                rows.append(dict(round=r,phase='train',timestamp=f'2026-01-01_00-00-{r*2:02d}',device_name=dev,pred_train_time_self='nan' if nan else 1.9,actual_train_time_self=2))
                rows.append(dict(round=r,phase='evaluate',timestamp=f'2026-01-01_00-00-{r*2:02d}',device_name=dev))
        if duplicate: rows.append(dict(rows[-4]))
        a.dump_csv(root/'fedavg.csv',rows)
        start=a.parse_server_ts('2026-01-01_00-00-00','UTC')
        for d in ['RPi5','jetson_orin_nx']:
            a.dump_csv(root/(d+'.csv'),[dict(unix_time_s=start+x*.5,power_w=3) for x in range(25)])
        return argparse.Namespace(fedavg=str(root/'fedavg.csv'),power=[d+'='+str(root/(d+'.csv')) for d in ['RPi5','jetson_orin_nx']],timezone='UTC',warmup=2,rounds=3,margin=1.1,max_gap=1,out=str(root/'slo.json'),slo=str(root/'slo.json'),outdir=str(root/'report'),time_error_limit=.1,time_accuracy_target=.9,energy_violation_target=.1,accuracy_metric='within_error_limit')
    def test_calibrate_and_score(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d)); a.lock_slo(args); self.assertEqual(a.score(args),2)
            report=json.loads((Path(d)/'report/tta_summary.json').read_text())
            self.assertAlmostEqual(report['overall']['mare'],.05)
            self.assertEqual(report['overall']['observations'],6)
            self.assertEqual(report['audit_status'],'not_audited')
            with self.assertRaises(FileExistsError): a.lock_slo(args)
    def test_duplicates_and_nan_rejected(self):
        for flag in ['duplicate','nan']:
            with tempfile.TemporaryDirectory() as d:
                args=self.fixture(Path(d),**{flag:True})
                if flag=='duplicate':
                    with self.assertRaises(ValueError): a.lock_slo(args)
                else:
                    a.lock_slo(args)
                    with self.assertRaises(ValueError): a.score(args)
    def test_slo_violation_is_fail(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d)); a.lock_slo(args)
            lock=json.loads(Path(args.slo).read_text()); lock['slo_joules']['RPi5']=1; Path(args.slo).write_text(json.dumps(lock))
            self.assertEqual(a.score(args),2)
    def test_reversed_power_is_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'power.csv'; a.dump_csv(p,[dict(unix_time_s=2,power_w=1),dict(unix_time_s=1,power_w=1)])
            with self.assertRaises(ValueError): a.read_power(p)

class ControlTests(unittest.TestCase):
    def worker(self,argv,timeout):
        temp=tempfile.TemporaryDirectory(); root=Path(temp.name)
        c.save(root/'test.job.json',dict(argv=argv,cwd=str(root),timeout_s=timeout))
        p=subprocess.Popen([sys.executable,str(c.HERE/'remote_worker.py'),str(root/'test.job.json')])
        return temp,root,p
    def test_remote_deadline(self):
        temp,root,p=self.worker([sys.executable,'-c','import time; time.sleep(20)'],.3)
        try:
            p.wait(12); self.assertEqual(c.read(root/'test.state.json')['status'],'timeout')
        finally:
            if p.poll() is None:p.kill()
            temp.cleanup()
    def test_remote_success_and_stop(self):
        temp,root,p=self.worker([sys.executable,'-c','print("ok")'],5)
        try: p.wait(12); self.assertEqual(c.read(root/'test.state.json')['status'],'success')
        finally: temp.cleanup()
    def test_stop_before_remote_launch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); marker=root/'child_was_started'
            c.save(root/'test.job.json',dict(argv=[sys.executable,'-c',f'open({str(marker)!r},"w").close()'],cwd=str(root),timeout_s=5))
            (root/'test.stop').touch()
            subprocess.run([sys.executable,str(c.HERE/'remote_worker.py'),str(root/'test.job.json')],check=True,timeout=5)
            self.assertEqual(c.read(root/'test.state.json')['status'],'stopped_before_start')
            self.assertFalse(marker.exists())
    def test_cleanup_with_never_started_job(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg={'server':{'alias':'server'},'hosts':{}}
            c.save(root/'state.json',{'remote_dirs':{'server':'/tmp/attempt'},'jobs':{'server':['superlink']}})
            completed=subprocess.CompletedProcess([],0,'','')
            with patch.object(c,'ssh',return_value=completed),patch.object(c,'status',return_value={'status':'not_started'}),patch.object(c,'collect'):
                self.assertEqual(c.cleanup(cfg,root),[])
    def test_status_distinguishes_ssh_failure_from_missing_job(self):
        transient=subprocess.CompletedProcess([],255,'','connection reset')
        running=subprocess.CompletedProcess([],0,'{"status":"running"}','')
        with patch.object(c,'ssh',side_effect=[transient,running]),patch.object(c.time,'sleep'):
            self.assertEqual(c.status({'alias':'jetson'},'/tmp/attempt','power')['status'],'running')
        absent=subprocess.CompletedProcess([],17,'','')
        with patch.object(c,'ssh',return_value=absent):
            self.assertEqual(c.status({'alias':'jetson'},'/tmp/attempt','power')['status'],'not_started')
        with patch.object(c,'ssh',return_value=transient),patch.object(c.time,'sleep'):
            with self.assertRaises(c.Failure): c.status({'alias':'jetson'},'/tmp/attempt','power')
    def test_jetson_control_address_preserves_verified_host_key(self):
        cfg=c.tomllib.loads((c.HERE/'suite.toml').read_text())
        args=c.ssh_target(cfg['hosts']['jetson'])
        self.assertIn('HostName=192.168.128.89',args)
        self.assertIn('HostKeyAlias=192.168.128.88',args)
        self.assertEqual(args[args.index('-J')+1],'pi5_5')
        self.assertEqual(args[-1],'jetson')
    def test_short_warmup_restores_measured_workload(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'TTA_OUTPUT_DIR':d}):
            spec=importlib.util.spec_from_file_location('tta_instrumentation_test',c.HERE/'instrumentation.py')
            module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            baseline={'max_samples_per_round':8,'rpi-normal-quota':8,
                      'rpi-correction-max-quota':8,'jetson-base-quota':8,
                      'jetson-max-quota':1024}
            settings={'warmup_rounds':2,'warmup_quota':2}
            for round_number in (1,2):
                adjusted=module.warmup_config(round_number,baseline,settings)
                self.assertEqual({adjusted[k] for k in baseline},{2})
                self.assertIsNot(adjusted,baseline)
            measured=module.warmup_config(3,baseline,settings)
            self.assertIs(measured,baseline)
            self.assertEqual(baseline['rpi-normal-quota'],8)
            self.assertEqual(baseline['jetson-max-quota'],1024)
            mixed=module.warmup_config(1,baseline,{'warmup_rounds':2,
                              'warmup_quota':6,'warmup_jetson_quota':8})
            self.assertEqual(mixed['rpi-normal-quota'],6)
            self.assertEqual(mixed['rpi-correction-max-quota'],6)
            self.assertEqual(mixed['jetson-base-quota'],8)
            self.assertEqual(mixed['jetson-max-quota'],8)
            self.assertEqual(mixed['max_samples_per_round'],8)
    def test_warmup_identity_unknown_but_measured_predictions_unique(self):
        records=[{'round':1,'device_name':None},{'round':1,'device_name':None},
                 {'round':2,'device_name':'RPi5'},{'round':2,'device_name':'jetson_orin_nx'},
                 {'round':3,'device_name':'RPi5'},{'round':3,'device_name':'jetson_orin_nx'}]
        index=c.measured_predictions(records,2,3,['RPi5','jetson_orin_nx'])
        self.assertEqual(len(index),2)
        with self.assertRaises(c.Failure): c.measured_predictions(records+[records[-1]],2,3,['RPi5','jetson_orin_nx'])
        with self.assertRaises(c.Failure): c.measured_predictions(records[:-1],2,3,['RPi5','jetson_orin_nx'])
    def test_diagnosis_prioritizes_primary_failure(self):
        hints=c.diagnosis('duplicate pre-dispatch prediction','Connection refused; power failure')
        self.assertEqual(len(hints),1)
        self.assertIn('prediction identity',hints[0])
        hints=c.diagnosis('jetson: cannot confirm remote attempt directory','power failure')
        self.assertEqual(len(hints),1)
        self.assertIn('SSH control connection',hints[0])
        hints=c.diagnosis('Wi-Fi TCP probe wlP1p1s0->192.168.128.124:22 failed; neighbor=FAILED')
        self.assertEqual(len(hints),1)
        self.assertIn('neighbor discovery',hints[0])
    def test_isolated_arrayrecord_patch(self):
        with tempfile.TemporaryDirectory() as d:
            app=Path(d); (app/'e2fl').mkdir()
            source=c.HERE/'evidence/server_client_app_llm.py'
            target=app/'e2fl/client_app_llm.py'; target.write_bytes(source.read_bytes())
            task_source=c.HERE/'evidence/server_task_llm.py'
            task_target=app/'e2fl/task_llm.py'; task_target.write_bytes(task_source.read_bytes())
            legacy_source=c.HERE/'evidence/server_task.py'
            legacy_target=app/'e2fl/task.py'; legacy_target.write_bytes(legacy_source.read_bytes())
            patch_file=c.HERE/'evidence/flower_arrayrecord_compatibility.patch'
            subprocess.run(['git','apply','--check',str(patch_file)],cwd=app,check=True)
            subprocess.run(['git','apply',str(patch_file)],cwd=app,check=True)
            self.assertIn('arrays_rec.to_torch_state_dict()',target.read_text())
            self.assertIn('    from fvcore.nn import FlopCountAnalysis',task_target.read_text())
            self.assertIn('    from fvcore.nn import FlopCountAnalysis',legacy_target.read_text())
            self.assertEqual(source.read_bytes(),(c.HERE/'evidence/server_client_app_llm.py').read_bytes())
            self.assertEqual(task_source.read_bytes(),(c.HERE/'evidence/server_task_llm.py').read_bytes())
            self.assertEqual(legacy_source.read_bytes(),(c.HERE/'evidence/server_task.py').read_bytes())
    def test_checksum_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg=dict(profile_common={},server={},hosts={},suite={})
            c.save(root/'slo_lock.json',{'test_fingerprint':c.fingerprint(cfg)})
            c.package(root); c.verify_reference(cfg,root)
            (root/'slo_lock.json').write_text('{}')
            with self.assertRaises(c.Failure):c.verify_reference(cfg,root)
    def test_resume_does_not_recalibrate_onsite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config=root/'c.toml'; config.write_text('[suite]\n')
            cfg={'suite':{}}; c.save(root/'intent.json',dict(stage='official',reference='/fixed/reference',fingerprint='f'))
            with patch.object(sys,'argv',['tta','-c',str(config),'resume','--run-dir',str(root)]),patch.object(c,'validate_reuse_config'),patch.object(c,'fingerprint',return_value='f'),patch.object(c,'trial',return_value=(0,root)) as trial:
                self.assertEqual(c.main(),0);self.assertEqual(trial.call_args.args[2],'official');self.assertEqual(trial.call_args.args[3],'/fixed/reference')

if __name__=='__main__':unittest.main()
