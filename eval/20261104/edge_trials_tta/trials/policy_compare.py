#!/usr/bin/env python3
"""Isolated Pi 5/NX policy comparison; never issues a TTA PASS/FAIL.

Uses the previously verified Exposure controller in a disposable Flower app.
All three modes use the same sampler, manifest, model, and seed. The paper
checkout and signed TTA runs are read only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import socket
import sys
import tarfile
import time
import tomllib
import uuid
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / 'harness'
sys.path.insert(0, str(HARNESS))
import orchestrator as core
from network_route import prepare as prepare_wifi_route, restore as restore_wifi_route

SOURCE_APP = ROOT / 'exposure_policy/physical_pi5_app'
DEFAULT_MANIFEST = SOURCE_APP / 'manifests/alpaca-cleaned/loracast_iid.json'
BASE_SUITE = HARNESS / 'suite.toml'
RUNS = Path(__file__).resolve().parent / 'policy_runs'
MODES = ('uniform', 'loracast-off', 'loracast-on')


def profile(args, manifest):
    if args.mode not in MODES:
        raise ValueError('unknown comparison mode')
    if args.rounds < 3 or args.rounds > 30:
        raise ValueError('comparison needs 2 warm-up and 1–28 evaluation rounds')
    if args.pi_quota < 1 or args.jetson_max < 2:
        raise ValueError('positive Pi quota and Jetson maximum >= 2 are required')
    exposure = getattr(args, 'exposure_max', None)
    cap = args.tbm_cap_samples
    if cap is not None and cap < 1:
        raise ValueError('--tbm-cap-samples must be positive')
    if exposure is not None and (not math.isfinite(exposure) or exposure <= 0):
        raise ValueError('--exposure-max must be a positive finite ratio')
    if args.mode == 'loracast-on' and (cap is None) == (exposure is None):
        raise ValueError('loracast-on requires exactly one of --exposure-max or --tbm-cap-samples')
    if args.mode != 'loracast-on' and (cap is not None or exposure is not None):
        raise ValueError('TBM limit applies only to loracast-on')
    if manifest.get('dataset_name') != args.dataset or manifest.get('num_clients') != 2:
        raise ValueError('manifest dataset or client count differs from requested condition')
    clients = manifest.get('clients', {})
    eligible = len(clients.get('0', {}).get('train_indices', []))
    if eligible < args.jetson_max:
        raise ValueError('NX manifest partition is smaller than the proposed quota')
    ratio = str(exposure) if exposure is not None else ('none' if cap is None else str((cap + 0.25) / eligible))
    if cap is not None and math.floor(float(ratio) * eligible + 1e-9) != cap:
        raise ValueError('TBM sample limit cannot be represented exactly')
    nx_base = args.pi_quota if args.mode == 'uniform' else getattr(args,'jetson_base',2)
    nx_max = nx_base if args.mode == 'uniform' else args.jetson_max
    nx_step=getattr(args,'jetson_step',2)
    if nx_base<1 or nx_base>nx_max or nx_step<1:
        raise ValueError('Jetson base must be positive and <= maximum; step must be positive')
    return {
        'strategy':'FinalExposureLoRACast',
        'sample-selection-mode':'exposure_aware',
        'aggregation-mode':'processed_samples',
        'rebalancing-enabled':False,
        'num-server-rounds':args.rounds,
        'fraction-train':1.0,'min-clients':2,'num_partitions':2,
        'max-samples-per-round':args.pi_quota,
        'batch_size':1,'local-epochs':1,'max-seq-len':args.max_seq_len,
        'enable-rpi-only-rounds':True,'use-token-ratio':True,
        'policy-warmup-rounds':2,
        'rpi-normal-quota':args.pi_quota,
        'rpi-sample-step':args.pi_quota,
        'rpi-correction-max-quota':args.pi_quota,
        'jetson-base-quota':nx_base,'jetson-max-quota':nx_max,
        'jetson-max-steps':nx_max,'jetson-sample-step':nx_step,
        'jetson-target-margin':0.95,
        'nx-eligible-unique':eligible,
        'nx-minimum-samples':nx_base,'nx-maximum-samples':nx_max,
        'nx-maximum-epochs':1,'exposure-max':ratio,
        'use-comm-prediction':False,
        'residual-calibration-enabled':True,
        'round-cooldown-enabled':False,
        'power-monitor':False,'resume':False,
        'save-every-round':1,
        'model':args.model,'dataset':args.dataset,
        'lora-r':16,'lora-r-RPi5':16,'lora-r-jetson_orin_nx':16,
        'lora-target-modules':'q_proj,k_proj,v_proj,o_proj',
        'sample-seed':args.seed,'experiment-seed':args.seed,
        'experiment-output-root':'',
        'device-profiles-path':'',
    }


def render_pyproject(source, overrides):
    original=source.read_text()
    head='[tool.flwr.app.config]'
    if original.count(head) != 1:
        raise ValueError('unexpected Flower config section')
    start=original.index(head)+len(head)
    next_section=re.search(r'(?m)^\[',original[start:])
    stop=start+next_section.start() if next_section else len(original)
    defaults=tomllib.loads(original)['tool']['flwr']['app']['config']
    missing=set(overrides)-set(defaults)
    if missing:
        raise ValueError(f'Flower defaults lack required policy controls: {sorted(missing)}')
    defaults.update(overrides)
    body='\n'+'\n'.join(json.dumps(k)+' = '+core.toml_scalar(v)
                        for k,v in defaults.items())+'\n'
    rendered=original[:start]+body+original[stop:]
    federation='[tool.flwr.federations.local-deployment]'
    if rendered.count(federation)!=1:
        raise ValueError('expected isolated Flower deployment config is missing')
    pos=rendered.index(federation)+len(federation)
    tail=rendered[pos:]
    tail,count=re.subn(r'(?m)^address[ \t]*=[^\n]*$',
                       'address = "127.0.0.1:19093"',tail,count=1)
    if count!=1:
        raise ValueError('Flower deployment address is missing')
    rendered=rendered[:pos]+tail
    if tomllib.loads(rendered)['tool']['flwr']['app']['config'] != defaults:
        raise ValueError('Flower config serialization failed')
    return rendered


def patch_output_paths(app):
    """Only redirect log files in the disposable copy into this attempt."""
    server=app/'e2fl/server_app_loracast.py'
    client=app/'e2fl/client_app_loracast.py'
    s=server.read_text(); c=client.read_text()
    needle='cfg(context, "experiment-output-root", "") or'
    if s.count(needle)!=2:
        raise ValueError('server output hook changed')
    s=s.replace(needle,needle+'\n        os.environ.get("TTA_OUTPUT_DIR") or')
    needle='context.run_config.get("experiment-output-root") or os.path.join('
    if c.count(needle)!=1:
        raise ValueError('client output hook changed')
    c=c.replace(needle,'context.run_config.get("experiment-output-root") or '
                'os.environ.get("TTA_OUTPUT_DIR") or os.path.join(')
    server.write_text(s);client.write_text(c)


def snapshot_app(run, overrides, manifest_path):
    app=run/'app'
    shutil.copytree(SOURCE_APP,app,ignore=shutil.ignore_patterns('__pycache__','*.pyc','eval','dry_import','test_*.py'))
    before={str(p.relative_to(app)):core.digest(p) for p in app.rglob('*') if p.is_file()}
    core.save(run/'source_hashes.json',before)
    patch_output_paths(app)
    shutil.copy2(Path(__file__).with_name('policy_prediction_audit.py'),app/'policy_prediction_audit.py')
    server=app/'e2fl/server_app_loracast.py'
    body=server.read_text();anchor='    strategy = create_strategy(context)'
    if body.count(anchor)!=1:raise ValueError('prediction audit insertion point changed')
    server.write_text(body.replace(anchor,anchor+'\n    from policy_prediction_audit import install\n    install(strategy)'))
    (app/'pyproject.toml').write_text(render_pyproject(app/'pyproject.toml',overrides))
    shutil.copy2(manifest_path,run/'partition_manifest.json')
    core.save(run/'staged_app_hashes.json',
              {str(p.relative_to(app)):core.digest(p) for p in app.rglob('*') if p.is_file()})
    archive=run/'app.tar.gz'
    with tarfile.open(archive,'w:gz') as t:t.add(app,arcname='app')
    return archive


def new_run(args, manifest_path, overrides):
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    run=RUNS/f'{stamp}_{args.mode}_{uuid.uuid4().hex[:6]}'
    run.mkdir(parents=True)
    overrides=overrides.copy()
    overrides['experiment-id']=run.name
    # The same manifest path exists on all three hosts during this attempt.
    overrides['partition-manifest-path']='/tmp/'+run.name+'_manifest.json'
    core.save(run/'condition.json',{'schema':1,'mode':args.mode,'model':args.model,
        'dataset':args.dataset,'rounds':args.rounds,'warmup_rounds':2,
        'seed':args.seed,'tbm_cap_samples':args.tbm_cap_samples,
        'exposure_max':getattr(args,'exposure_max',None),
        'manifest_sha256':core.digest(manifest_path),'config':overrides,
        'purpose':'exploratory workload/TBM comparison; never a TTA PASS'})
    return run,overrides


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def summarize(run):
    """Summarize observed training only; missing replies remain missing."""
    run=Path(run)
    cond=core.read(run/'condition.json')
    root=run/'server/output'/cond['config']['experiment-id']/'server'
    fed=list(root.glob('fedavg_*.csv'))
    if len(fed)!=1:
        raise ValueError('exactly one server training CSV is required')
    rows=[r for r in read_rows(fed[0]) if r.get('phase')=='train']
    aggregation=read_rows(root/'aggregation.csv')
    by_round={}
    for row in rows:
        r=int(row['round']);dev=row['device_name']
        if dev in by_round.setdefault(r,{}):
            raise ValueError(f'duplicate {dev} training row in round {r}')
        by_round[r][dev]=row
    if set(by_round)!=set(range(1,cond['rounds']+1)):
        raise ValueError('incomplete observed round sequence')
    report=[]
    for r in range(1,cond['rounds']+1):
        devs=by_round[r]
        if 'RPi5' not in devs:
            raise ValueError(f'Pi reply absent in round {r}')
        decision_file=root/f'final_exposure_decision_round_{r:03d}.json'
        decision=core.read(decision_file) if decision_file.exists() else {}
        if 'jetson_orin_nx' not in devs and decision.get('action')!='omit_before_training':
            raise ValueError(f'NX reply absent without an explicit TBM omission in round {r}')
        weights={x['device_uid']:x['aggregation_weight'] for x in aggregation
                 if int(x['round'])==r}
        for dev,row in devs.items():
            report.append({'mode':cond['mode'],'round':r,'segment':'warmup' if r<=2 else 'evaluation',
                'device_name':dev,'selected_samples':int(row['num_examples']),
                'input_tokens':int(float(row.get('input_tokens') or 0)),
                'local_steps':int(float(row.get('local_steps') or 0)),
                'actual_train_time_s':float(row.get('actual_train_time_self') or 0),
                'actual_upload_time_s':float(row.get('actual_upload_time_self') or 0),
                'aggregation_weight':weights.get(row['device_uid'],''),
                'round_mode':row.get('round_mode',''),
                'tbm_action':decision.get('action','normal'),
                'tbm_remaining_before':decision.get('remaining_before',''),
                'tbm_cap_samples':cond['tbm_cap_samples'] or '',
                'exposure_max':cond.get('exposure_max') or '',
                'source_csv':str(fed[0].relative_to(run))})
        if 'jetson_orin_nx' not in devs:
            report.append({'mode':cond['mode'],'round':r,'segment':'warmup' if r<=2 else 'evaluation',
                'device_name':'jetson_orin_nx','selected_samples':0,'input_tokens':0,
                'local_steps':0,'actual_train_time_s':'','actual_upload_time_s':'',
                'aggregation_weight':0,'round_mode':'PROTECTED_ONLY',
                'tbm_action':'omit_before_training','tbm_remaining_before':decision.get('remaining_before',''),
                'tbm_cap_samples':cond['tbm_cap_samples'] or '',
                'exposure_max':cond.get('exposure_max') or '',
                'source_csv':str(fed[0].relative_to(run))})
    path=run/'policy_rounds.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(report[0]));writer.writeheader();writer.writerows(report)
    core.save(run/'policy_summary.json',{'schema':1,'status':'observed',
        'mode':cond['mode'],'model':cond['model'],'dataset':cond['dataset'],
        'manifest_sha256':cond['manifest_sha256'],'rounds':cond['rounds'],
        'warmup_rounds':2,'tbm_cap_samples':cond['tbm_cap_samples'],
        'exposure_max':cond.get('exposure_max'),
        'pi_eval_samples':sum(x['selected_samples'] for x in report if x['device_name']=='RPi5' and x['segment']=='evaluation'),
        'nx_eval_samples':sum(x['selected_samples'] for x in report if x['device_name']=='jetson_orin_nx' and x['segment']=='evaluation'),
        'nx_omitted_rounds':[x['round'] for x in report if x['device_name']=='jetson_orin_nx' and x['tbm_action']=='omit_before_training'],
        'tta_pass':None,'limitation':'Exploratory policy control; TTA six-prediction and energy-SLO criteria are not applied.'})
    return path


def run_hardware(args):
    manifest_path=Path(args.manifest).resolve()
    manifest=json.loads(manifest_path.read_text())
    overrides=profile(args,manifest)
    cfg=tomllib.loads(BASE_SUITE.read_text())
    # Long runs must not hit the short suite's 35-minute worker deadline.
    cfg['suite']['attempt_timeout_s']=max(cfg['suite']['attempt_timeout_s'],args.rounds*150+600)
    run,overrides=new_run(args,manifest_path,overrides)
    print('Run:',run,flush=True)
    core.save(run/'state.json',{'status':'starting','stage':'policy-comparison',
        'remote_dirs':{},'jobs':{},'start':time.time(),
        'attempt_timeout_s':cfg['suite']['attempt_timeout_s']})
    manifest_remote=overrides['partition-manifest-path']
    try:
        prepare_wifi_route()
        pf=core.preflight(cfg,run)
        if not all(pf['hosts'][name]['checks'].get('route_is_wifi') for name in cfg['hosts']):
            raise ValueError('comparison requires direct Wi-Fi FL traffic')
        archive=snapshot_app(run,overrides,manifest_path)
        state=core.read(run/'state.json')
        for name,h in core.all_hosts(cfg).items():
            root=core.remote_root(h,run);lock=root.rsplit('/',1)[0]+'/active.lock'
            state['remote_dirs'][name]=root;state['jobs'][name]=[];core.save(run/'state.json',state)
            core.ssh(h,'mkdir -p '+core.q(root)+' && mkdir '+core.q(lock)+
                     ' && printf %s '+core.q(root)+' > '+core.q(lock+'/owner'))
            core.ssh(h,'mkdir -p '+core.q(root+'/output'))
            core.upload(h,HARNESS/'remote_worker.py',root+'/remote_worker.py')
            core.upload(h,manifest_path,manifest_remote)
            core.upload(h,archive,root+'/app.tar.gz')
            core.ssh(h,'tar -xzf '+core.q(root+'/app.tar.gz')+' -C '+core.q(root))
            probe=('import e2fl.server_app_loracast' if name=='server'
                   else 'import e2fl.client_app_loracast')
            command=('env PYTHONPATH='+core.q(root+'/app')+' TTA_OUTPUT_DIR='+core.q(root+'/output')+
                     ' timeout -k 3 90 '+core.q(h['python'])+' -c '+core.q(probe))
            core.ssh(h,command,timeout=100)
        deadline=cfg['suite']['attempt_timeout_s']
        def launch(name,job,argv,cwd=None):
            h=core.all_hosts(cfg)[name];root=state['remote_dirs'][name]
            state['jobs'][name].append(job);core.save(run/'state.json',state)
            core.start(h,root,job,argv,cwd or root,deadline,run)
        for name,h in cfg['hosts'].items():
            root=state['remote_dirs'][name]
            core.upload(h,HARNESS/'power_logger.py',root+'/power_logger.py')
            launch(name,'power',[h['python'],root+'/power_logger.py','--device',h['sensor'],
                  '--output',root+'/power.csv','--meta',root+'/power.meta.json',
                  '--interval',str(cfg['suite']['power_sample_interval_s'])])
        server_root=state['remote_dirs']['server']
        launch('server','superlink',[cfg['server']['bin']+'/flower-superlink','--insecure',
            '--fleet-api-address','0.0.0.0:19092','--control-api-address','127.0.0.1:19093',
            '--serverappio-api-address','127.0.0.1:19091'])
        time.sleep(3)
        fleet=cfg['server']['fleet_address'];host,port=fleet.rsplit(':',1)
        for name,h in cfg['hosts'].items():
            core.py(h,'import socket; s=socket.create_connection('+repr((host,int(port)))+',timeout=5); s.close()',20)
            root=state['remote_dirs'][name]
            launch(name,'supernode',[h['bin']+'/flower-supernode','--insecure','--superlink',fleet,
                '--clientappio-api-address','127.0.0.1:'+str(h['client_port']),
                '--node-config',f'partition-id={h["partition"]} num-partitions=2'])
        launch('server','flwr',[cfg['server']['bin']+'/flwr','run','.','local-deployment','--stream'],server_root+'/app')
        state['status']='running';core.save(run/'state.json',state)
        started=time.monotonic()
        while True:
            st=core.status(cfg['server'],server_root,'flwr')
            if st['status'] not in ('running','starting'):
                if st['status']!='success':raise RuntimeError('Flower failed: '+json.dumps(st))
                break
            for name,jobs in state['jobs'].items():
                for job in jobs:
                    if job!='flwr' and core.status(core.all_hosts(cfg)[name],state['remote_dirs'][name],job)['status']!='running':
                        raise RuntimeError(f'{name}/{job} stopped during comparison')
            if time.monotonic()-started>deadline:raise RuntimeError('comparison deadline exceeded')
            if int(time.monotonic()-started)%30<3:
                print(f'[{args.mode}] running {time.monotonic()-started:.0f}s',flush=True)
            time.sleep(3)
        state['status']='executed';core.save(run/'state.json',state)
    except BaseException as exc:
        state=core.read(run/'state.json');state.update(status='invalid',error=str(exc));core.save(run/'state.json',state)
        print('INVALID:',exc,file=sys.stderr,flush=True)
    finally:
        state=core.read(run/'state.json');state['end']=time.time();core.save(run/'state.json',state)
        if state.get('remote_dirs'):
            errors=core.cleanup(cfg,run)
            if errors:print('CLEANUP:',errors,file=sys.stderr)
        for name,h in core.all_hosts(cfg).items():
            if name in state.get('remote_dirs',{}):
                try:core.ssh(h,'rm -f '+core.q(manifest_remote),check=False)
                except Exception:pass
        try:restore_wifi_route()
        except Exception as exc:print('ROUTE CLEANUP:',exc,file=sys.stderr)
    if core.read(run/'state.json')['status']=='executed':
        try:
            path=summarize(run)
            state=core.read(run/'state.json');state['status']='complete';core.save(run/'state.json',state)
            print('Observed policy CSV:',path)
        except Exception as exc:
            state=core.read(run/'state.json');state.update(status='invalid',error=str(exc));core.save(run/'state.json',state)
            print('INVALID ANALYSIS:',exc,file=sys.stderr)
    core.package(run)
    if core.read(run/'state.json')['status']=='complete':
        show_report(run,getattr(args,'no_open',False))
    return 0 if core.read(run/'state.json')['status']=='complete' else 2


def show_report(run,no_open=False):
    from policy_compare_report import single_report
    page=single_report(run)
    print('결과 화면:',page,flush=True)
    if not no_open:
        webbrowser.open(page.as_uri())
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description='Isolated Gemma/Pi5/NX workload and Exposure TBM comparison')
    p.add_argument('--mode',choices=MODES)
    p.add_argument('--show-run',type=Path,help='학습 없이 기존 실행의 웹 결과 생성 및 열기')
    p.add_argument('--no-open',action='store_true',help='HTML은 저장하고 브라우저 자동 열기는 생략')
    p.add_argument('--model',default='hf:google/gemma-3-270m')
    p.add_argument('--dataset',default='yahma/alpaca-cleaned')
    p.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--seed',type=int,default=47)
    p.add_argument('--pi-quota',type=int,default=2)
    p.add_argument('--jetson-max',type=int,default=16)
    p.add_argument('--jetson-base',type=int,default=2)
    p.add_argument('--jetson-step',type=int,default=2)
    p.add_argument('--max-seq-len',type=int,default=64,help='tokenizer 길이 상한 (기존 실장비 비교 기준: 64)')
    p.add_argument('--tbm-cap-samples',type=int)
    p.add_argument('--exposure-max',type=float,help='NX total sample visits / eligible unique samples (paper Exposure-2: 2)')
    p.add_argument('--dry-run',action='store_true',help='validate and print intended controls without contacting hosts')
    args=p.parse_args(argv)
    try:
        if args.show_run:
            if args.mode or args.dry_run:
                raise ValueError('--show-run은 --mode 또는 --dry-run과 함께 사용할 수 없습니다.')
            return show_report(args.show_run,args.no_open)
        if not args.mode:
            raise ValueError('실행에는 --mode, 기존 결과 보기에는 --show-run이 필요합니다.')
        manifest=json.loads(args.manifest.read_text())
        config=profile(args,manifest)
        if args.dry_run:
            print(json.dumps({'mode':args.mode,'model':args.model,'dataset':args.dataset,
                'manifest_sha256':core.digest(args.manifest),'config':config,
                'warning':'exploratory control comparison; TTA PASS/FAIL is not calculated'},
                indent=2,ensure_ascii=False))
            return 0
        return run_hardware(args)
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        print('ERROR:',exc,file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
