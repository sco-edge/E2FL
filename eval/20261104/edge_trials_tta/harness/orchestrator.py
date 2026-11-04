#!/usr/bin/env python3
"""Mac controller for isolated SSH LoRACast trials (Python 3.11+, stdlib)."""
import argparse, concurrent.futures, hashlib, json, os, shlex, shutil, subprocess
import sys, tarfile, time, tomllib, uuid, socket
from datetime import datetime
from pathlib import Path
import analyze as analyze
import audit_evidence
import render_report
import predictor_state

HERE = Path(__file__).resolve().parent
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2']
q = shlex.quote
class Failure(RuntimeError): pass

def ssh_target(h):
    args=list(SSH)
    if h.get('jump_host'):
        args += ['-J',h['jump_host']]
    if h.get('control_host'):
        args += ['-o', 'HostName='+h['control_host'], '-o', 'HostKeyAlias='+h['host_key_alias']]
    return args+[h['alias']]

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)); tmp.replace(path)
def read(path): return json.loads(Path(path).read_text())
def ssh(h, command, timeout=45, check=True, binary=False):
    p = subprocess.run(ssh_target(h) + [command], capture_output=True, text=not binary, timeout=timeout)
    if check and p.returncode: raise Failure(f"{h['alias']}: exit {p.returncode}: {p.stderr!s}\n{p.stdout!s}")
    return p

def upload(h, src, dst):
    with Path(src).open('rb') as f:
        p = subprocess.run(ssh_target(h) + ['cat > '+q(dst)], stdin=f, capture_output=True, timeout=60)
    if p.returncode: raise Failure(f'upload {h["alias"]}: {p.stderr!r}')

def env(h, out=None):
    e = {'PATH':h['bin']+':/usr/local/bin:/usr/bin:/bin', 'PYTHONUNBUFFERED':'1', 'TZ':'Asia/Seoul', 'TTA_PMIC_SUDO':'1' if h.get('pmic_sudo') else '0'}
    if out: e['TTA_OUTPUT_DIR'] = out
    return e

def py(h, code, timeout=45):
    return ssh(h, 'timeout -k 3 '+str(timeout-5)+' '+q(h['python'])+' -c '+q(code), timeout=timeout)

def all_hosts(cfg): return {'server':cfg['server'], **cfg['hosts']}

def environment_signature(pf):
    return {name: {'imports':json.loads(info['checks']['imports']),
                   'sensor':info['checks'].get('power',{}).get('device'),
                   'sensor_paths':info['checks'].get('power',{}).get('paths'),
                   'route':info['checks'].get('route_to_server'),
                   'wifi_link':info['checks'].get('wifi_link'),
                   'wifi_power_save':info['checks'].get('wifi_power_save'),
                   'wifi_11ax_disabled':info['checks'].get('wifi_11ax_disabled')}
            for name,info in pf['hosts'].items()}

def parse_wifi_link(output):
    """Keep the SSID, AP, and band that define a calibration's radio path."""
    link={}
    for line in output.splitlines():
        line=line.strip()
        if line.startswith('Connected to '): link['bssid']=line.split()[2].lower()
        elif line.startswith('SSID:'): link['ssid']=line.partition(':')[2].strip()
        elif line.startswith('freq:'): link['frequency_mhz']=int(float(line.partition(':')[2].strip()))
    if set(link)!={'bssid','ssid','frequency_mhz'}:
        raise Failure('Wi-Fi link is disconnected or its AP/band cannot be identified')
    return link

def jetson_wifi_health(cfg):
    """Read transport and firmware state without changing the device."""
    h=cfg['hosts']['jetson']
    destination=cfg['server']['fleet_address'].rsplit(':',1)[0]
    route=json.loads(ssh(h,'ip -j route get '+q(destination)).stdout)[0]
    interface='wlP1p1s0'
    crashes=ssh(h,"sudo -n journalctl -k -b --no-pager | grep -c 'Microcode SW error detected' || true").stdout.strip()
    return {'observed_at':time.time(), 'boot_id':ssh(h,'cat /proc/sys/kernel/random/boot_id').stdout.strip(),
            'route':route, 'wifi_power_save':ssh(h,'iw dev '+q(interface)+' get power_save').stdout.strip(),
            'wifi_link':parse_wifi_link(ssh(h,'iw dev '+q(interface)+' link').stdout),
            'wifi_11ax_disabled':ssh(h,'cat /sys/module/iwlwifi/parameters/disable_11ax').stdout.strip(),
            'firmware_crash_count':int(crashes or 0)}

def validate_jetson_wifi_health(before,after):
    if before['boot_id']!=after['boot_id']:
        raise Failure('Jetson rebooted during the trial')
    if after['firmware_crash_count']!=before['firmware_crash_count']:
        raise Failure('Jetson iwlwifi firmware crashed during the trial')
    if before['route'].get('dev')!=after['route'].get('dev') or before['route'].get('prefsrc')!=after['route'].get('prefsrc'):
        raise Failure('Jetson server route changed during the trial')
    if before['wifi_link']!=after['wifi_link']:
        raise Failure('Jetson Wi-Fi SSID, AP, or band changed during the trial')
    if before['wifi_power_save']!=after['wifi_power_save'] or before['wifi_11ax_disabled']!=after['wifi_11ax_disabled']:
        raise Failure('Jetson Wi-Fi driver settings changed during the trial')

def audit_relative_clock(preflight, post_clock, names, tolerance):
    """Mac wall-clock adjustments cancel out of server/client alignment."""
    for name in names:
        pre_server=preflight['hosts']['server']['clock']; pre_client=preflight['hosts'][name]['clock']
        after_server=post_clock['server']; after_client=post_clock[name]
        pre_relative=pre_client['offset_s']-pre_server['offset_s']
        after_relative=after_client['offset_s']-after_server['offset_s']
        uncertainty=max(pre_client['uncertainty_s']+pre_server['uncertainty_s'],
                        after_client['uncertainty_s']+after_server['uncertainty_s'])
        after_client['relative_drift_bound_s']=abs(after_relative-pre_relative)+uncertainty
        after_client['server_alignment_bound_s']=abs(after_relative)+after_client['uncertainty_s']+after_server['uncertainty_s']
    if any(post_clock[name]['relative_drift_bound_s']>tolerance or
           post_clock[name]['server_alignment_bound_s']>tolerance for name in names):
        raise Failure('client/server relative clock drift exceeds test tolerance')
    return post_clock

def fingerprint(cfg):
    return canonical(dict(profile=cfg['profile_common'], hosts=all_hosts(cfg), suite=cfg['suite'],
                          harness={p.name:digest(p) for p in HERE.glob('*.py') if not p.name.startswith('test')},
                          compatibility_patch=digest(HERE/'evidence/flower_arrayrecord_compatibility.patch'),
                          alpha_correction_patch=digest(HERE/'evidence/online_time_scale_correction.patch')))

def validate_reuse_config(cfg):
    suite=cfg['suite']; profile=cfg['profile_common']; stages=cfg['profiles']
    prime_slo=suite.get('slo_source')=='prime'
    expected_policy='two_disjoint_windows' if prime_slo else 'three_disjoint_windows'
    if suite.get('sample_split_policy') != expected_policy:
        raise Failure('TTA sample split does not match the selected SLO source')
    if suite.get('token_input_mode','historical') not in ('historical','predeclared_diagnostic'):
        raise Failure('token_input_mode must be historical or predeclared_diagnostic')
    cap=int(profile['max-seq-len'])
    if cap < 1 or any(int(stages[name].get('max-seq-len',cap)) != cap
                      for name in ('prime','calibration','official')):
        raise Failure('TTA sequence cap must be positive and identical in prime, calibration and official')
    expected={'prime':2,'official':3}
    if not prime_slo: expected['calibration']=5
    if (suite['warmup_rounds']!=0
            or any(stages[name]['num-server-rounds']!=rounds for name,rounds in expected.items())
            or suite['calibration_measurement_rounds'] != (2 if prime_slo else 5)):
        raise Failure('state-reuse round counts do not match the selected SLO source')
    if (int(profile['rpi-normal-quota'])!=2 or int(profile['jetson-base-quota'])!=2
            or int(profile['jetson-max-quota'])!=2 or float(profile['participation-fairness-rho'])!=0
            or int(suite.get('warmup_quota',0))!=0):
        raise Failure('state-reuse trial is validated only for the fixed quota-2 profile')

def diagnosis(text,logs=''):
    tests = [('duplicate pre-dispatch prediction', 'Inspect measured-round prediction identity/count; warm-up identity may be unknown'),
             ('Node Unavailable', 'A sampled client lost its Flower heartbeat; inspect Wi-Fi neighbor state, AP forwarding, and the client SuperNode log before retrying'),
             ('missing/duplicate/unexpected device-round training rows', 'Inspect server/flwr.log for rejected replies or a client heartbeat loss; preserve this attempt and retry only after connectivity recovers'),
             ('Wi-Fi TCP probe', 'Direct client-to-server Wi-Fi path failed; check both radios, AP forwarding and neighbor discovery before restarting the trial'),
             ('cannot confirm remote attempt directory', 'SSH control connection dropped during cleanup; reconnect, then resume so owned jobs and leases are collected'),
             ('SSH relay disconnected', 'SSH control tunnel dropped; inspect local relay logs and the control link'),
             ('remote port forwarding failed', 'Reverse tunnel port is occupied; use an attempt-specific relay port'),
             ('cannot read process status', 'SSH status probe failed; verify the host link before retrying'),
             ('non-tensor LoRA', 'Existing client discarded server LoRA tensors; payload decoding must be corrected before valid FL testing'),
             ('TTA_INVALID_PAYLOAD', 'Empty server adapter payload; original client ArrayRecord conversion is invalid'),
             ('Permission denied','SSH authentication or sensor permissions; inspect which command failed'),
             ('timed out','SSH/import timeout; check host reachability, storage I/O and the selected Python environment'),
             ('exit 124','Remote command timed out; inspect import and storage health'),
             ('Segmentation','Python/native dependency crash; use the verified environment, do not reinstall blindly'),
             ('No module named','Missing dependency in the configured environment'),
             ('CUDA out of memory','Workload exceeds available GPU memory'),
             ('Address already in use','Port occupied; leave unrelated processes alone'),
             ('Connection refused','SuperLink is not reachable from a client'),
             ('GatedRepo','Model cache/access is not ready'),
             ('clock','Clock alignment is outside the test tolerance'),
             ('power','Sensor/trace failure; energy result is invalid')]
    primary=[hint for token,hint in tests if token.lower() in text.lower()]
    if primary: return primary
    secondary=[hint for token,hint in tests if token.lower() in logs.lower()]
    return secondary or ['Inspect the complete per-host logs; no automatic code or workload change was applied.']

def package(run):
    run = Path(run)
    files = [p for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ('checksums.json','checksums.sha256')]
    (run/'checksums.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(run)}\n' for p in files))
    hashes = {str(p.relative_to(run)):digest(p) for p in files+[run/'checksums.sha256']}
    save(run/'checksums.json', hashes)
    dest = run.with_suffix('.tar.gz')
    with tarfile.open(dest, 'w:gz') as t: t.add(run, arcname=run.name)
    return str(dest)

def preflight(cfg, dest):
    """Gather all hosts even if one fails. No dependency installation or daemon changes."""
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    def inspect(item):
        name,h = item; result = {'alias':h['alias'], 'checks':{}, 'errors':[]}
        probe = '/tmp/loracast-probe-'+uuid.uuid4().hex+'.py'
        checks = {
          'system': 'import json,platform,shutil,os,time; print(json.dumps(dict(host=platform.node(),python=platform.python_version(),disk_free=shutil.disk_usage("/tmp").free,unix=time.time())))',
          'imports': 'import json,torch,flwr,transformers,peft,datasets; print(json.dumps(dict(flwr=flwr.__version__,torch=torch.__version__,cuda=torch.cuda.is_available(),transformers=transformers.__version__)))',
        }
        for key,code in checks.items():
            try: result['checks'][key] = py(h,code,cfg['suite'].get('import_timeout_s',180) if key=='imports' else 20).stdout.strip()
            except Exception as e: result['errors'].append(f'{key}: {e}')
        try:
            samples=[]
            for _ in range(3):
                a=time.time(); remote_t=float(py(h,'import time; print(time.time())',20).stdout); b=time.time()
                samples.append({'offset_s':remote_t-(a+b)/2, 'uncertainty_s':(b-a)/2})
            result['clock']=min(samples,key=lambda x:x['uncertainty_s'])
            ports = [19091,19092,19093] if name=='server' else [h['client_port']]
            result['checks']['ports']=py(h,'import socket\n'+ '\n'.join(f's=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(("0.0.0.0",{p})); s.close()' for p in ports),20).returncode
            if name!='server':
                destination=cfg['server']['fleet_address'].rsplit(':',1)[0]
                route=ssh(h,'ip -j route get '+q(destination)).stdout
                result['checks']['route_to_server']=json.loads(route)
                interface=result['checks']['route_to_server'][0].get('dev','')
                result['checks']['route_is_wifi']=ssh(h,'test -d '+q('/sys/class/net/'+interface+'/wireless'),check=False).returncode==0
                if result['checks']['route_is_wifi']:
                    result['checks']['wifi_link']=parse_wifi_link(ssh(h,'iw dev '+q(interface)+' link',timeout=10).stdout)
                    result['checks']['wifi_power_save']=ssh(h,'iw dev '+q(interface)+' get power_save',timeout=10).stdout.strip()
                    if name=='jetson':
                        result['checks']['wifi_11ax_disabled']=ssh(h,'cat /sys/module/iwlwifi/parameters/disable_11ax',timeout=10).stdout.strip()
                    wifi_check=('import socket; s=socket.socket(); '
                                's.setsockopt(socket.SOL_SOCKET,socket.SO_BINDTODEVICE,'+repr((interface+'\0').encode())+'); '
                                's.settimeout(3); s.connect(('+repr(destination)+',22)); s.close()')
                    try: py(h,wifi_check,12)
                    except Exception as e:
                        try: neighbor=ssh(h,'ip neigh show '+q(destination)+' dev '+q(interface),timeout=10,check=False).stdout.strip()
                        except Exception: neighbor='unavailable'
                        raise Failure(f'Wi-Fi TCP probe {interface}->{destination}:22 failed; neighbor={neighbor or "unknown"}: {e}') from e
                    result['checks']['wifi_tcp22']=True
                upload(h,HERE/'power_logger.py',probe)
                command='env '+ ' '.join(k+'='+q(v) for k,v in env(h).items())+' '+q(h['python'])+' '+q(probe)+' --device '+h['sensor']+' --probe'
                result['checks']['power']=json.loads(ssh(h,command,20).stdout)
        except Exception as e: result['errors'].append(str(e))
        finally:
            try: ssh(h,'rm -f '+q(probe),check=False)
            except Exception: pass
        result['ok']=not result['errors']; return name,result
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=dict(pool.map(inspect,all_hosts(cfg).items()))
    server_clock=results['server'].get('clock')
    for name,result in results.items():
        clock=result.get('clock')
        if clock and server_clock and name!='server':
            bound=abs(clock['offset_s']-server_clock['offset_s'])+clock['uncertainty_s']+server_clock['uncertainty_s']
            result['server_alignment_bound_s']=bound
            if bound > cfg['suite']['max_clock_offset_ms']/1000:
                result['errors'].append('clock alignment to server exceeds tolerance')
                result['ok']=False
    versions=[json.loads(r['checks']['imports'])['flwr'] for r in results.values() if 'imports' in r['checks']]
    if len(set(versions))>1:
        results['server']['errors'].append('Flower versions differ across hosts'); results['server']['ok']=False
    data={'created':time.time(),'hosts':results,'ok':all(x['ok'] for x in results.values())}
    save(dest/'preflight.json',data)
    if not data['ok']:
        errors='\n'.join(f'{n}: {e}' for n,r in results.items() for e in r['errors'])
        save(dest/'diagnosis.json',{'error':errors,'suggestions':diagnosis(errors)})
        raise Failure(errors)
    return data

def snapshot(cfg, run):
    """Read a small source snapshot from the server; never modify its repository."""
    h=cfg['server']
    code='''import io,tarfile,pathlib,sys
root=pathlib.Path(REPO)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for name in ['e2fl','log','power','pyproject.toml']:
  p=root/name
  files=[p] if p.is_file() else sorted(p.rglob('*.py'))
  for f in files:
   if f.is_file() and '__pycache__' not in f.parts: t.add(f,arcname=str(f.relative_to(root)))
'''.replace('REPO',repr(h['repo']))
    data=ssh(h,q(h['python'])+' -c '+q(code),binary=True).stdout
    archive=run/'source.tar.gz'; archive.write_bytes(data)
    app=run/'app'; app.mkdir()
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            target=(app/member.name).resolve()
            if not target.is_relative_to(app.resolve()) or not member.isfile(): raise Failure('unsafe source archive')
        t.extractall(app,filter='data')
    save(run/'source_hashes.json',{str(p.relative_to(app)):digest(p) for p in app.rglob('*') if p.is_file()})
    commit=ssh(h,'git -C '+q(h['repo'])+' rev-parse HEAD',check=False)
    save(run/'source_commit.json',{'commit':commit.stdout.strip() if commit.returncode==0 else None})
    patch = HERE/'evidence/flower_arrayrecord_compatibility.patch'
    for flag in ('--check', None):
        command = ['git', 'apply'] + ([flag] if flag else []) + [str(patch)]
        result = subprocess.run(command, cwd=app, text=True, capture_output=True)
        if result.returncode:
            raise Failure(f'isolated ArrayRecord compatibility patch failed: {result.stderr}')
    save(run/'compatibility_patch.json', {'patch_sha256': digest(patch),
                                           'patched_client_sha256': digest(app/'e2fl/client_app_llm.py'),
                                           'patched_task_sha256': digest(app/'e2fl/task_llm.py'),
                                           'patched_legacy_task_sha256': digest(app/'e2fl/task.py')})
    alpha_source = app/'e2fl/strategy/FedAvgLoRA.py'
    alpha_before = digest(alpha_source)
    alpha_patch = HERE/'evidence/online_time_scale_correction.patch'
    for flag in ('--check', None):
        command = ['git', 'apply'] + ([flag] if flag else []) + [str(alpha_patch)]
        result = subprocess.run(command, cwd=app, text=True, capture_output=True)
        if result.returncode:
            raise Failure(f'isolated alpha correction patch failed: {result.stderr}')
    save(run/'alpha_correction_patch.json', {
        'source_sha256': alpha_before,
        'patch_sha256': digest(alpha_patch),
        'patched_source_sha256': digest(alpha_source),
        'research_checkout_modified': False,
    })
    # Add observation-only hooks to the disposable app copy. The server's
    # source snapshot and research checkout are left untouched.
    client = app/'e2fl/client_app_llm.py'
    body = client.read_text()
    edits = (
        ('from e2fl.task_llm import set_weights, test, train as train_model, get_flops, is_lora_model, get_peft_model_state_dict',
         'from e2fl.task_llm import set_weights, test, train as train_model, get_flops, is_lora_model, get_peft_model_state_dict\nfrom sample_audit import AuditSubset, event as tta_event, adapter_modules as tta_adapter_modules\nfrom run_settings import SAMPLE_OFFSET'),
        ('    try:\n        partition_id, num_partitions = get_partition_id_from_ip(context)\n    except ValueError as e:\n        logger.warning(f"{e} → fallback to partition_id=0")\n        partition_id, num_partitions = 0, 1',
         '    # TTA protocol: use the partition explicitly assigned by Flower.\n    node_config = context.node_config\n    partition_id = int(node_config["partition-id"])\n    num_partitions = int(node_config["num-partitions"])\n    if num_partitions != 2 or partition_id not in (0, 1):\n        raise RuntimeError("TTA requires two distinct assigned data partitions")'),
        ('        "dataset_name": dataset_name,',
         '        "dataset_name": dataset_name,\n        "partition_id": partition_id,'),
        ('indices = list(range(num))',
         'if len(dataset) < SAMPLE_OFFSET + num:\n                raise RuntimeError("TTA sample window exceeds data partition")\n            indices = list(range(SAMPLE_OFFSET, SAMPLE_OFFSET + num))'),
        ('subset = Subset(dataset, indices)',
         'subset = AuditSubset(dataset, indices, server_round, dev_name, device_uid, _LOCAL_STATE["partition_id"])'),
        ('            trainloader = base_trainloader\n\n    # Determine how many examples are actually used for this round',
         '            raise RuntimeError("TTA strict data selection failed") from e\n\n    # Determine how many examples are actually used for this round'),
        ('    t_loop0 = time.perf_counter()\n    train_result = train_model(',
         '    tta_event(server_round, dev_name, device_uid, _LOCAL_STATE["partition_id"], "start", workload={\n        "max_samples_per_round": max_samples_per_round,\n        "selected_samples": num_examples_used,\n        "local_steps": int(_LOCAL_STATE.get("local_steps", 0) or 0),\n        "local_epochs": int(_LOCAL_STATE["local_epochs"]),\n        "batch_size": int(trainloader.batch_size),\n        "max_seq_len_configured": get_cfg("max_seq_len", None),\n        "sample_offset": SAMPLE_OFFSET,\n        "lora_rank": _LOCAL_STATE.get("lora_rank"),\n        "lora_modules_configured": get_cfg("lora_modules", None),\n        "active_lora_modules": tta_adapter_modules(_LOCAL_STATE["net"]),\n        "model": _LOCAL_STATE.get("model_name"),\n        "dataset": _LOCAL_STATE.get("dataset_name"),\n    })\n    t_loop0 = time.perf_counter()\n    train_result = train_model('),
        ('    t_loop1 = time.perf_counter()\n    loop_time = t_loop1 - t_loop0',
         '    t_loop1 = time.perf_counter()\n    tta_event(server_round, dev_name, device_uid, _LOCAL_STATE["partition_id"], "end")\n    loop_time = t_loop1 - t_loop0'),
    )
    for before,after in edits:
        if body.count(before)!=1:
            raise Failure('client audit insertion point changed: '+before[:55])
        body=body.replace(before,after)
    client.write_text(body)
    dataset=app/'e2fl/dataset.py'
    data_body=dataset.read_text()
    start_marker='    # 안전하게 partition_id 로드: out-of-range 예외를 잡아 모듈로로 보정'
    end_marker='    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)'
    if data_body.count(start_marker)!=1 or data_body.count(end_marker)!=1:
        raise Failure('dataset partition fallback insertion point changed')
    start=data_body.index(start_marker); end=data_body.index(end_marker,start)
    data_body=data_body[:start]+'    # TTA data independence requires the exact assigned partition.\n    partition = fds.load_partition(partition_id)\n'+data_body[end:]
    data_body=enforce_sequence_cap(data_body)
    dataset.write_text(data_body)
    save(run/'isolated_audit_patch.json',{'client_sha256':digest(client),
                                          'dataset_sha256':digest(dataset),
                                          'partition_fallback':'disabled in disposable TTA app'})
    for name in ['instrumentation.py','run_settings.py','predictor_state.py','server_entry.py','client_entry.py','sample_audit.py']:
        shutil.copy2(HERE/name,app/name)
    return app

def enforce_sequence_cap(dataset_source):
    """Apply the configured tokenizer limit only in the disposable TTA app."""
    anchor='from e2fl.task import is_hf_model_name'
    tokenizer_args='truncation=True, padding=False'
    if dataset_source.count(anchor)!=1 or dataset_source.count(tokenizer_args)!=2:
        raise Failure('dataset tokenizer insertion points changed')
    return dataset_source.replace(anchor,
        anchor+'\nfrom run_settings import TTA_MAX_SEQ_LEN').replace(
        tokenizer_args,'truncation=True, max_length=TTA_MAX_SEQ_LEN, padding=False')

def toml_scalar(x):
    if isinstance(x,bool): return str(x).lower()
    return json.dumps(x)

def sample_offset(stage, cap, policy='three_disjoint_windows'):
    if cap < 1:
        raise Failure('sample cap must be positive for the phase split')
    if policy=='two_disjoint_windows':
        return {'smoke':0, 'prime':cap, 'official':2*cap}[stage]
    return {'smoke':0, 'prime':0, 'calibration':cap, 'official':2*cap}[stage]

def calibration_official_protocol_valid(calibration, official):
    """Require identical test conditions and distinct, fixed sample windows."""
    if not isinstance(calibration,dict) or not isinstance(official,dict):
        return False
    cap=calibration.get('sample_count_cap')
    if not isinstance(cap,int) or cap < 1 or official.get('sample_count_cap') != cap:
        return False
    if (calibration.get('sample_split_policy') not in ('three_disjoint_windows','two_disjoint_windows')
            or official.get('sample_split_policy') != calibration.get('sample_split_policy')
            or calibration.get('sample_offset') != cap
            or official.get('sample_offset') != 2*cap):
        return False
    return ({k:v for k,v in calibration.items() if k != 'sample_offset'}
            == {k:v for k,v in official.items() if k != 'sample_offset'})

def prepare_app(cfg,app,profile,remote_dir,state_mode='off',state_file=None,
                planned_features=None,sample_offset_value=0):
    original=tomllib.loads((app/'pyproject.toml').read_text())
    defaults=original['tool']['flwr']['app']['config']
    defaults.update(profile)
    warmup_quota=int(cfg['suite'].get('warmup_quota',0))
    warmup_jetson_quota=int(cfg['suite'].get('warmup_jetson_quota',warmup_quota))
    if warmup_quota:
        if warmup_quota<1 or warmup_quota>int(cfg['profile_common']['rpi-normal-quota']):
            raise Failure('warmup quota must be between 1 and the measured RPi quota')
        if warmup_jetson_quota<1 or warmup_jetson_quota>int(cfg['profile_common']['jetson-base-quota']):
            raise Failure('warmup Jetson quota must be between 1 and the measured Jetson base quota')
        defaults['tta-warmup-rounds']=int(cfg['suite']['warmup_rounds'])
        defaults['tta-warmup-quota']=warmup_quota
        defaults['tta-warmup-jetson-quota']=warmup_jetson_quota
    # server_app_llm selects run_config fields explicitly. A Python module is
    # included in the Flower app archive and read by the isolated wrapper.
    (app/'run_settings.py').write_text(
        f'WARMUP_ROUNDS = {int(cfg["suite"]["warmup_rounds"])}\n'
        f'WARMUP_QUOTA = {warmup_quota}\n'
        f'WARMUP_JETSON_QUOTA = {warmup_jetson_quota}\n'
        f'STATE_MODE = {state_mode!r}\n'
        f'SAMPLE_OFFSET = {int(sample_offset_value)}\n'
        f'PLANNED_SAMPLE_OFFSET = {2*int(cfg["profile_common"]["rpi-normal-quota"])}\n'
        f'PLANNED_SAMPLE_COUNT = {int(cfg["profile_common"]["rpi-normal-quota"]) if cfg["suite"].get("token_input_mode", "historical") == "predeclared_diagnostic" else 0}\n'
        f'PLANNED_TOKENS_PER_EXAMPLE = '
        f'{({name: item["tokens_per_example"] for name,item in planned_features["devices"].items()} if planned_features else {})!r}\n'
        f'TTA_MAX_SEQ_LEN = {int(cfg["profile_common"]["max-seq-len"])}\n'
        f'STATE_JSON = {(Path(state_file).read_text() if state_file else "")!r}\n')
    if state_mode == 'reuse':
        if state_file is None: raise Failure('predictor state is required for reuse')
        predictor_state.read(state_file)
    defaults['checkpoint-path']=remote_dir+'/output/checkpoints'
    defaults['resume']=False
    # No stale device profile is silently inherited from an unrelated experiment.
    defaults['device-profiles-path']=''
    body='''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[project]
name = "loracast-tta"
version = "1.0.0"
description = "Isolated instrumentation of existing LoRACast"
license = "Apache-2.0"
[tool.flwr.app]
publisher = "wwjang"
[tool.flwr.app.components]
serverapp = "server_entry:app"
clientapp = "client_entry:app"
[tool.flwr.app.config]
'''
    body+='\n'.join(qkey+' = '+toml_scalar(v) for k,v in defaults.items() for qkey in [json.dumps(k)])
    body+='\n[tool.flwr.federations]\ndefault = "tta"\n[tool.flwr.federations.tta]\ninsecure = true\naddress = '+json.dumps(cfg['server']['control_address'])+'\n'
    (app/'pyproject.toml').write_text(body)
    return defaults

def remote_root(h, run):
    home=ssh(h,'printf %s "$HOME"').stdout
    return home+'/.loracast-tta/'+run.name

def start(h,root,name,argv,cwd,timeout,local):
    job={'argv':argv,'cwd':cwd,'timeout_s':timeout,'env':env(h,root+'/output')}
    file=local/(name+'.job.json'); save(file,job); upload(h,file,root+'/'+file.name)
    ssh(h,'nohup '+q(h['python'])+' '+q(root+'/remote_worker.py')+' '+q(root+'/'+file.name)+' > '+q(root+'/'+name+'.supervisor.log')+' 2>&1 < /dev/null &')
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        state=status(h,root,name)
        if state['status']=='running': return
        if state['status']!='not_started': raise Failure(f'{h["alias"]}/{name} exited during launch: {state}')
        time.sleep(.2)
    raise Failure(f'{h["alias"]}/{name} did not create a process state')

def status(h,root,name):
    path=root+'/'+name+'.state.json'
    last_error=None
    for attempt in range(4):
        try:
            p=ssh(h,'if test -e '+q(path)+'; then cat '+q(path)+'; else exit 17; fi',check=False)
            if p.returncode==0: return json.loads(p.stdout)
            if p.returncode==17:
                return {'status':'not_started'}
            last_error=f'exit {p.returncode}: {p.stderr}'
        except subprocess.TimeoutExpired as e:
            last_error=str(e)
        if attempt<3: time.sleep(.5)
    raise Failure(f'{h["alias"]}/{name}: cannot read process status after retries: {last_error}')

def collect(h,root,local):
    # Only this attempt's outputs; no broad repository globs or checkpoints.
    code='''import pathlib,tarfile,sys
r=pathlib.Path(ROOT)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for p in sorted(r.rglob('*')):
  if p.is_file() and p.suffix in ('.log','.json','.jsonl','.csv','.txt') and 'app' not in p.relative_to(r).parts:
   t.add(p,arcname=str(p.relative_to(r)))
'''.replace('ROOT',repr(root))
    data=ssh(h,q(h['python'])+' -c '+q(code),timeout=90,binary=True).stdout
    local.mkdir(exist_ok=True,parents=True); archive=local/'remote.tar.gz'; archive.write_bytes(data)
    with tarfile.open(archive) as t: t.extractall(local,filter='data')

def cleanup(cfg,run):
    state=read(run/'state.json'); errors=[]
    for name,root in state.get('remote_dirs',{}).items():
        h=all_hosts(cfg)[name]; jobs=state.get('jobs',{}).get(name,[])
        try:
            exists=ssh(h,'test -d '+q(root),check=False)
            if exists.returncode==1:
                continue
            if exists.returncode!=0:
                raise Failure(f'{name}: cannot confirm remote attempt directory')
            for job in jobs: ssh(h,'touch '+q(root+'/'+job+'.stop'))
            deadline=time.monotonic()+20
            while any(status(h,root,j)['status']=='running' for j in jobs):
                if time.monotonic()>deadline: raise Failure('cleanup not confirmed; remote lease retained')
                time.sleep(.5)
            collect(h,root,run/name)
            # Release only the lease owned by this exact attempt.
            lock=root.rsplit('/',1)[0]+'/active.lock'
            cmd='test "$(cat '+q(lock+'/owner')+' 2>/dev/null)" = '+q(root)+' && rm '+q(lock+'/owner')+' && rmdir '+q(lock)
            unlock=ssh(h,cmd,check=False)
            if unlock.returncode and ssh(h,'test -d '+q(lock),check=False).returncode==0:
                raise Failure(f'{name}: lease belongs to another attempt or could not be released')
        except Exception as e: errors.append(name+': '+str(e))
    for job in state.get('local_jobs',[]):
        try:
            root=run/'local'; (root/(job+'.stop')).touch()
            deadline=time.monotonic()+20
            while (root/(job+'.state.json')).exists() and read(root/(job+'.state.json'))['status']=='running':
                if time.monotonic()>deadline: raise Failure('local SSH relay cleanup timeout')
                time.sleep(.2)
        except Exception as e: errors.append('local '+job+': '+str(e))
    state['cleanup_errors']=errors; save(run/'state.json',state)
    return errors

def execute(cfg,run,stage,pf):
    state={'status':'starting','stage':stage,'fingerprint':fingerprint(cfg),'remote_dirs':{},'jobs':{},'start':time.time()}
    save(run/'state.json',state)
    try:
        for name,h in all_hosts(cfg).items():
            root=remote_root(h,run); lock=root.rsplit('/',1)[0]+'/active.lock'
            state['remote_dirs'][name]=root; state['jobs'][name]=[]; save(run/'state.json',state)
            ssh(h,'mkdir -p '+q(root)+' && mkdir '+q(lock)+' && printf %s '+q(root)+' > '+q(lock+'/owner'))
            ssh(h,'mkdir -p '+q(root+'/output/eval'))
            upload(h,HERE/'remote_worker.py',root+'/remote_worker.py')
        if (stage!='smoke' or cfg['suite'].get('smoke_transport')=='direct') and cfg['suite'].get('require_wifi',True):
            for name in cfg['hosts']:
                if not pf['hosts'][name]['checks'].get('route_is_wifi'):
                    raise Failure(name+': server route is not Wi-Fi; official/calibration transport precondition failed')
        app=snapshot(cfg,run)
        profile={**cfg['profile_common'],**cfg['profiles'][stage]}
        server_root=state['remote_dirs']['server']
        if stage == 'official' and read(run/'source_hashes.json') != read(run/'reference/source_hashes.json'):
            raise Failure('source changed since SLO calibration')
        state_file=None
        if stage in ('calibration','official'):
            source=(run/'state_reference' if stage=='calibration' else
                    run/'reference' if read(run/'reference/slo_lock.json').get('slo_source')=='prime'
                    else run/'reference/state_reference')
            manifest=read(source/'predictor_manifest.json')
            if manifest['source_hashes']!=read(run/'source_hashes.json'):
                raise Failure('source changed since predictor priming')
            if manifest['environment_signature']!=environment_signature(pf):
                raise Failure('environment changed since predictor priming')
            state_file=source/'predictor_state.json'
            if digest(state_file)!=manifest['state_sha256']:
                raise Failure('predictor state hash mismatch')
            if stage=='official' and manifest['state_sha256']!=read(run/'reference/slo_lock.json')['state_sha256']:
                raise Failure('SLO was calibrated from a different predictor state')
        planned_features = None
        if cfg['suite'].get('token_input_mode','historical') == 'predeclared_diagnostic':
            if stage == 'calibration':
                planned_features = audit_evidence.build_planned_feature_manifest(run/'state_reference',cfg)
            elif stage == 'official':
                planned_features = audit_evidence.build_planned_feature_manifest(
                    run/'reference' if read(run/'reference/slo_lock.json').get('slo_source')=='prime'
                    else run/'reference/state_reference',cfg)
                if planned_features != read(run/'reference/slo_lock.json').get('planned_input_features'):
                    raise Failure('planned sample inputs differ from the SLO-locked pre-training manifest')
        if planned_features:
            save(run/'planned_input_features.json',planned_features)
        cap=int(cfg['profile_common']['rpi-normal-quota'])
        offset=sample_offset(stage,cap,cfg['suite']['sample_split_policy'])
        full_profile=prepare_app(cfg,app,profile,server_root,
                                 'prime' if stage=='prime' else 'reuse' if state_file else 'off',
                                 state_file,planned_features,offset)
        save(run/'effective_config.json',full_profile)
        save(run/'sample_protocol.json',{'partition_source':'Flower context.node_config',
              'partition_map':{name:h['partition'] for name,h in cfg['hosts'].items()},
              'token_input_mode':cfg['suite'].get('token_input_mode','historical'),
              'max_seq_len':int(cfg['profile_common']['max-seq-len']),
              'sequence_length_cap_enforced':True,
              'sample_split_policy':cfg['suite']['sample_split_policy'],
              'sample_offset':offset,
              'sample_count_cap':cap})
        with tarfile.open(run/'app.tar.gz','w:gz') as t: t.add(app,arcname='app')
        upload(cfg['server'],run/'app.tar.gz',server_root+'/app.tar.gz')
        ssh(cfg['server'],'tar -xzf '+q(server_root+'/app.tar.gz')+' -C '+q(server_root))
        # Validate each client's actual Python environment against the patched
        # app before starting power sampling or a Flower round.
        for name,h in cfg['hosts'].items():
            root=state['remote_dirs'][name]
            upload(h,run/'app.tar.gz',root+'/app.tar.gz')
            ssh(h,'tar -xzf '+q(root+'/app.tar.gz')+' -C '+q(root))
            import_timeout=int(cfg['suite'].get('import_timeout_s',180))
            check_cmd=('env PYTHONPATH='+q(root+'/app')+' TTA_OUTPUT_DIR='+q(root+'/output')
                       +' timeout -k 3 '+str(import_timeout-3)+' '+q(h['python'])
                       +' -c '+q('import client_entry; print("client_app_import_ok")'))
            probe=ssh(h,check_cmd,timeout=import_timeout+5)
            (run/(name+'_app_import.txt')).write_text(probe.stdout)
        deadline=cfg['suite']['attempt_timeout_s']
        def launch(name,job,argv,cwd=None):
            h=all_hosts(cfg)[name]; root=state['remote_dirs'][name]
            state['jobs'][name].append(job); save(run/'state.json',state)
            start(h,root,job,argv,cwd or root,deadline,run)
        for name,h in cfg['hosts'].items():
            root=state['remote_dirs'][name]; upload(h,HERE/'power_logger.py',root+'/power_logger.py')
            launch(name,'power',[h['python'],root+'/power_logger.py','--device',h['sensor'],'--output',root+'/power.csv','--meta',root+'/power.meta.json','--interval',str(cfg['suite']['power_sample_interval_s'])])
        h=cfg['server']
        launch('server','superlink',[h['bin']+'/flower-superlink','--insecure','--fleet-api-address','0.0.0.0:19092','--control-api-address','127.0.0.1:19093','--serverappio-api-address','127.0.0.1:19091'])
        time.sleep(3)
        if status(h,server_root,'superlink')['status']!='running': raise Failure('SuperLink failed to start')
        fleet_address=cfg['server']['fleet_address']
        state['transport']='direct'
        if stage=='smoke' and cfg['suite'].get('smoke_transport')=='ssh-relay':
            state['transport']='ssh-relay-functional-only'
            local=run/'local'; local.mkdir()
            state['local_jobs']=[]
            def local_launch(name,argv):
                save(local/(name+'.job.json'),{'argv':argv,'cwd':str(local),'timeout_s':deadline})
                state['local_jobs'].append(name); save(run/'state.json',state)
                with (local/(name+'.supervisor.log')).open('wb') as log:
                    subprocess.Popen([sys.executable,str(HERE/'remote_worker.py'),str(local/(name+'.job.json'))],stdout=log,stderr=log,start_new_session=True)
            sock=socket.socket(); sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]; sock.close()
            # A fixed reverse port can remain occupied while an earlier SSH
            # connection is timing out. Use a fresh port for each attempt.
            relay_port=20000+int(uuid.uuid4().hex[:8],16)%30000
            state['smoke_relay_port']=relay_port; save(run/'state.json',state)
            local_launch('relay-server',SSH+['-o','ExitOnForwardFailure=yes','-N','-L',f'127.0.0.1:{port}:127.0.0.1:19092',cfg['server']['alias']])
            for name,h in cfg['hosts'].items():
                local_launch('relay-'+name,ssh_target(h)[:-1]+['-o','ExitOnForwardFailure=yes','-N','-R',f'127.0.0.1:{relay_port}:127.0.0.1:{port}',h['alias']])
            time.sleep(3)
            for name in state['local_jobs']:
                if read(local/(name+'.state.json'))['status']!='running': raise Failure('SSH relay failed: '+name)
            fleet_address=f'127.0.0.1:{relay_port}'
        fleet_host,fleet_port=fleet_address.rsplit(':',1)
        for name,h in cfg['hosts'].items():
            py(h, 'import socket; s=socket.create_connection('+repr((fleet_host,int(fleet_port)))+',timeout=5); s.close()',20)
            launch(name,'supernode',[h['bin']+'/flower-supernode','--insecure','--superlink',fleet_address,'--clientappio-api-address','127.0.0.1:'+str(h['client_port']),'--node-config',f'partition-id={h["partition"]} num-partitions=2'])
        launch('server','flwr',[cfg['server']['bin']+'/flwr','run','.','tta','--stream'],server_root+'/app')
        state.update(status='running'); save(run/'state.json',state)
        start_time=time.monotonic(); last_print=0
        while True:
            s=status(cfg['server'],server_root,'flwr')
            if s['status'] not in ('running','starting'):
                if s['status']!='success': raise Failure('Flower run failed: '+json.dumps(s))
                break
            for name,jobs in state['jobs'].items():
                for job in jobs:
                    if job=='flwr': continue
                    stat=status(all_hosts(cfg)[name],state['remote_dirs'][name],job)
                    if stat['status']!='running': raise Failure(f'{name}/{job} stopped: {stat}')
            for job in state.get('local_jobs',[]):
                if read(run/'local'/(job+'.state.json'))['status']!='running': raise Failure('SSH relay disconnected: '+job)
            elapsed=time.monotonic()-start_time
            if elapsed>deadline: raise Failure('attempt deadline exceeded')
            if elapsed-last_print>30:
                try:
                    probe=ssh(cfg['server'],'cat '+q(server_root+'/output/predictions.jsonl'),timeout=10,check=False)
                    records=[json.loads(line) for line in probe.stdout.splitlines() if line.strip()] if probe.returncode==0 else []
                    save(run/'progress.json',{'rounds_predicted':sorted({int(x['round']) for x in records}),
                                              'observed_at':time.time()})
                except Exception:
                    pass
                print(f'[{stage}] running {elapsed:.0f}s: {run.name}',flush=True); last_print=elapsed
            time.sleep(3)
        post_clock={}
        for name,h in all_hosts(cfg).items():
            samples=[]
            for _ in range(3):
                a=time.time(); t=float(py(h,'import time; print(time.time())',20).stdout); b=time.time()
                samples.append({'offset_s':t-(a+b)/2,'uncertainty_s':(b-a)/2})
            post_clock[name]=min(samples,key=lambda x:x['uncertainty_s'])
        save(run/'post_clock.json',post_clock)
        audited=audit_relative_clock(pf,post_clock,cfg['hosts'],cfg['suite']['max_clock_offset_ms']/1000)
        save(run/'post_clock.json',audited)
        state['status']='executed'
    except BaseException as e:
        state.update(status='failed',error=str(e)); save(run/'diagnosis.json',{'error':str(e),'suggestions':diagnosis(str(e))})
        raise
    finally:
        state['end']=time.time(); save(run/'state.json',state)
        errors=cleanup(cfg,run)
        save(run/'timing.json',{'elapsed_s':time.time()-state['start'],'target_s':cfg['suite']['onsite_target_s']})
        package(run)
        if errors: raise Failure('; '.join(errors))
    normalize(cfg,run,stage,pf)
    prediction_log=run/'server/output/predictions.jsonl'
    if prediction_log.is_file():
        records=[json.loads(line) for line in prediction_log.read_text().splitlines() if line.strip()]
        save(run/'progress.json',{'rounds_predicted':sorted({int(x['round']) for x in records}),
                                  'observed_at':time.time(),'source':'collected_predictions.jsonl'})
    state=read(run/'state.json'); state['status']='validated'; save(run/'state.json',state)


def measured_predictions(predictions,warmup,rounds,devices):
    expected={(r,device) for r in range(warmup+1,rounds+1) for device in devices}
    index={}
    for record in predictions:
        if record['round']<=warmup:
            continue  # Device identity is not known before the first reply.
        key=(record['round'],record['device_name'])
        if key in index: raise Failure('duplicate pre-dispatch prediction')
        index[key]=record
    if set(index)!=expected:
        raise Failure('missing/unexpected measured-round pre-dispatch prediction')
    return index

def normalize(cfg,run,stage,pf):
    import csv, math
    server_log=run/'server/flwr.log'
    if server_log.is_file():
        log=server_log.read_text(errors='replace')
        if 'ServerApp raised an exception' in log or 'Exit Code: 201' in log:
            detail='Node Unavailable (client heartbeat lost)' if 'Node Unavailable' in log else 'see server/flwr.log'
            raise Failure('Flower ServerApp failed: '+detail)
    for logfile in run.rglob('supernode.log'):
        log=logfile.read_text(errors='replace')
        if 'non-tensor LoRA entries before set_weights' in log or 'TTA_INVALID_PAYLOAD' in log:
            raise Failure('Invalid server adapter decoding: '+str(logfile))
    candidates=list((run/'server/output').rglob('fedavg_*.csv'))
    if len(candidates)!=1: raise Failure(f'Expected exactly one isolated fedavg CSV; got {len(candidates)}')
    rows=analyze.read_fedavg(candidates[0]); n=cfg['profiles'][stage]['num-server-rounds']
    expected={(r,h['device']) for r in range(1,n+1) for h in cfg['hosts'].values()}
    train=[r for r in rows if r['phase']=='train']
    keys=[(int(r['round']),r['device_name']) for r in train]
    if set(keys)!=expected or len(keys)!=len(expected): raise Failure('missing/duplicate/unexpected device-round training rows')
    predictions=[json.loads(x) for x in (run/'server/output/predictions.jsonl').read_text().splitlines()]
    warmup=n if stage=='prime' else 2 if stage=='smoke' else cfg['suite']['warmup_rounds']
    if stage=='prime':
        index={(int(p['round']),str(p['cid'])):p for p in predictions}
        if len(index)!=len(predictions):
            raise Failure('duplicate prime pre-dispatch timestamp')
    elif warmup==0:
        index={}
        for p in predictions:
            key=(int(p['round']),str(p['cid']))
            if key in index: raise Failure('duplicate pre-dispatch prediction')
            index[key]=p
        if set(index)!={(int(r['round']),str(r['cid'])) for r in train}:
            raise Failure('missing/unexpected measured-round pre-dispatch prediction')
    else:
        index=measured_predictions(predictions,warmup,n,[h['device'] for h in cfg['hosts'].values()])
    server_offset=pf['hosts']['server']['clock']['offset_s']
    for row in train:
        r=int(row['round'])
        if stage=='prime':
            p=index.get((r,str(row['cid'])))
            if p is None: raise Failure('missing prime pre-dispatch timestamp')
            row['prediction_recorded_at']=p['predicted_at']
        if r<=warmup: continue
        p=index[(r,str(row['cid']))] if warmup==0 else index[(r,row['device_name'])]
        if p['device_name'] not in (None,row['device_name']):
            raise Failure('pre-dispatch device identity changed after reply')
        value=p['pred_train_time_s'] if p['device_name'] else p.get('candidates',{}).get(row['device_name'])
        actual=float(row['actual_train_time_self'])
        if value is None or not math.isfinite(value) or value<=0 or not math.isfinite(actual) or actual<=0: raise Failure('invalid prediction/actual time')
        if any(x>=r for x in p['history_rounds']): raise Failure('prediction uses current/future history')
        # Prefer absolute training start from existing metrics if available; emission
        # itself is guaranteed before configure_train returns any outbound messages.
        row['post_reply_prediction_original']=row['pred_train_time_self']
        row['pred_train_time_self']=value
        row['prediction_recorded_at']=p['predicted_at']
    analyze.dump_csv(run/'fedavg.csv',rows)
    for name,h in cfg['hosts'].items():
        source=run/name/'power.csv'; shift=server_offset-pf['hosts'][name]['clock']['offset_s']
        with source.open() as f: power=list(csv.DictReader(f))
        for row in power: row['unix_time_s']=float(row['unix_time_s'])+shift
        analyze.dump_csv(run/name/'power_aligned.csv',power)
    # Validate strict trace coverage, including interpolation's bracketing gaps.
    if stage!='prime' or cfg['suite'].get('slo_source')=='prime':
        energy_warmup=0 if stage=='prime' else warmup
        analyze.energy_observations(rows,{h['device']:run/name/'power_aligned.csv' for name,h in cfg['hosts'].items()},cfg['suite']['timezone'],energy_warmup,n-energy_warmup,cfg['suite']['max_power_sample_gap_s'])


def analysis_args(cfg,run,stage):
    from types import SimpleNamespace
    s=cfg['suite']
    return SimpleNamespace(fedavg=str(run/'fedavg.csv'),power=[h['device']+'='+str(run/name/'power_aligned.csv') for name,h in cfg['hosts'].items()],timezone=s['timezone'],warmup=s['warmup_rounds'],rounds=cfg['profiles'][stage]['num-server-rounds']-s['warmup_rounds'],margin=s['slo_margin'],max_gap=s['max_power_sample_gap_s'],clock_tolerance_s=s['max_clock_offset_ms']/1000,time_error_limit=s['time_error_limit'],time_accuracy_target=s['time_accuracy_target'],energy_violation_target=s['energy_violation_target'],accuracy_metric=s['accuracy_metric'],run_dir=str(run))

def record_prime_samples(run):
    for source_name,target_name in (('sample_audit.jsonl','prime_samples.jsonl'),
                                    ('planned_samples.jsonl','planned_samples.jsonl')):
        lines=[]
        for host in ('rpi5','jetson'):
            path=run/host/'output'/source_name
            if path.is_file(): lines += path.read_text().splitlines()
        if not lines and source_name == 'sample_audit.jsonl':
            raise Failure('missing prime '+source_name)
        if lines:
            (run/target_name).write_text('\n'.join(lines)+'\n')

def slo_audit(run,ref,at_start):
    lock=run/'reference/slo_lock.json'
    after=digest(lock)
    original=digest(ref/'slo_lock.json')
    body=read(lock)
    calculated={device:max(values)*float(body['margin'])
                for device,values in body['calibration_energy_j'].items()}
    formula_matches=all(abs(calculated[d]-float(body['slo_joules'][d]))<1e-8
                        for d in calculated)
    power_hosts={'RPi5':'rpi5','jetson_orin_nx':'jetson'}
    calibration_hashes_match=(body.get('calibration_raw_csv_sha256')==digest(ref/'fedavg.csv')
                              and body.get('tta_suite_sha256')==digest(ref/'suite.toml')
                              and all(body.get('power_trace_sha256',{}).get(dev)==digest(ref/host/'power_aligned.csv')
                                      for dev,host in power_hosts.items()))
    count=2 if body.get('slo_source')=='prime' else 5
    sample_lengths_match=True
    if count==2:
        import csv
        with (run/'report/tta_dataset_samples.csv').open(newline='') as stream:
            samples=list(csv.DictReader(stream))
        def lengths(phase,device):
            return sorted(int(x['input_token_count']) for x in samples
                          if x['phase']==phase and x['device_name']==device)
        cap=int(body['sample_protocol']['sample_count_cap'])
        sample_lengths_match=all(
            len(lengths('prime',dev))==len(lengths('official',dev))==cap
            and lengths('prime',dev)==lengths('official',dev)
            for dev in power_hosts)
    protocol_valid=(float(body.get('margin',0))==1.10
                    and body.get('calibration_round_numbers')==list(range(1,count+1))
                    and body.get('measurement_rounds')==count
                    and read(ref/'state.json').get('stage')==('prime' if count==2 else 'calibration')
                    and set(body.get('calibration_energy_j',{}))==set(power_hosts)
                    and all(len(values)==count for values in body['calibration_energy_j'].values())
                    and body.get('sample_protocol')==read(ref/'sample_protocol.json')
                    and sample_lengths_match
                    and calibration_official_protocol_valid(
                        body.get('sample_protocol'),read(run/'sample_protocol.json')))
    lock_preceded_official=datetime.fromisoformat(body['created_at']).timestamp() < float(read(run/'slo_start.json')['recorded_at'])
    result={'schema':1,'slo_sha256_before':at_start,'slo_sha256_after':after,
            'reference_slo_sha256_after':original,
            'unchanged':at_start==after==original,
            'formula':body['formula'],'margin':body['margin'],
            'calibration_round_numbers':body.get('calibration_round_numbers'),
            'calibration_round_count':body.get('measurement_rounds'),
            'calibration_energy_j':body['calibration_energy_j'],
            'workload_config':body.get('workload_config'),
            'sample_protocol':body.get('sample_protocol'),
            'slo_joules':body['slo_joules'],
            'created_at':body['created_at'],
            'calibration_raw_csv_sha256':body.get('calibration_raw_csv_sha256'),
            'power_trace_sha256':body.get('power_trace_sha256'),
            'tta_suite_sha256':body.get('tta_suite_sha256'),
            'formula_matches':formula_matches,
            'calibration_hashes_match':calibration_hashes_match,
            'calibration_protocol_valid':protocol_valid,
            'calibration_sample_lengths_match_official':sample_lengths_match,
            'lock_preceded_official':lock_preceded_official,
            'source':'reference/slo_lock.json',
            'official_results_used_in_calibration':not lock_preceded_official}
    audit_evidence.write_json(run/'report/tta_slo_audit.json',result)
    if not result['unchanged'] or not formula_matches or not calibration_hashes_match or not protocol_valid or not lock_preceded_official:
        raise Failure('SLO lock changed or calibration evidence mismatch during official run')
    return result

def new_run(config,stage,parent=None):
    d=HERE/'runs'/(time.strftime('%Y%m%d_%H%M%S')+'_'+stage+'_'+uuid.uuid4().hex[:6]); d.mkdir(parents=True)
    shutil.copy2(config,d/'suite.toml')
    (d/'harness').mkdir()
    for source in HERE.glob('*.py'): shutil.copy2(source,d/'harness'/source.name)
    shutil.copy2(HERE/'evidence/flower_arrayrecord_compatibility.patch',d/'harness'/ 'arrayrecord_compatibility.patch')
    shutil.copy2(HERE/'evidence/online_time_scale_correction.patch',d/'harness'/ 'alpha_correction.patch')
    if parent: save(d/'parent.json',{'previous_attempt':str(parent)})
    print('Run: '+str(d),flush=True); return d

def verify_reference(cfg,path):
    ref=Path(path).resolve(); lock=read(ref/'slo_lock.json'); hashes=read(ref/'checksums.json')
    for name,sha in hashes.items():
        p=(ref/name).resolve()
        if not p.is_relative_to(ref) or digest(p)!=sha: raise Failure('reference checksum mismatch: '+name)
    if lock['test_fingerprint']!=fingerprint(cfg): raise Failure('SLO reference/config/harness mismatch; recalibration required')
    return ref

def verify_prime(cfg,path):
    prime=Path(path).resolve(); manifest=read(prime/'predictor_manifest.json')
    for name,sha in read(prime/'checksums.json').items():
        target=(prime/name).resolve()
        if not target.is_relative_to(prime) or digest(target)!=sha:
            raise Failure('prime checksum mismatch: '+name)
    if manifest['test_fingerprint']!=fingerprint(cfg):
        raise Failure('predictor priming/config/harness mismatch')
    if digest(prime/'predictor_state.json')!=manifest['state_sha256']:
        raise Failure('predictor state hash mismatch')
    if cfg['suite'].get('token_input_mode','historical') == 'predeclared_diagnostic':
        if digest(prime/'planned_samples.jsonl')!=manifest.get('planned_samples_sha256'):
            raise Failure('pre-training sample feature hash mismatch')
        audit_evidence.build_planned_feature_manifest(prime,cfg)
    predictor_state.read(prime/'predictor_state.json')
    return prime

def trial(cfg,config,stage,reference=None,parent=None,state_reference=None):
    if stage=='calibration' and cfg['suite'].get('slo_source')=='prime':
        raise Failure('prime already sets the SLO; no separate calibration run is needed')
    ref=verify_reference(cfg,reference) if stage=='official' else None
    prime=verify_prime(cfg,state_reference) if stage=='calibration' else None
    if stage=='calibration' and prime is None:
        raise Failure('calibration requires a primed predictor state')
    retries=cfg['suite']['max_retries'] if stage!='official' else 0
    for attempt in range(retries+1):
        entry_time=time.time()
        run=new_run(config,stage,parent)
        save(run/'intent.json',{'stage':stage,'reference':str(ref) if ref else None,
                                'state_reference':str(prime) if prime else None,
                                'fingerprint':fingerprint(cfg)})
        if ref:
            shutil.copytree(ref,run/'reference',ignore=shutil.ignore_patterns('*.tar.gz','app','vendor'))
            slo_at_start=digest(run/'reference/slo_lock.json')
            save(run/'slo_start.json',{'slo_lock_sha256':slo_at_start,'recorded_at':time.time()})
        if prime:
            (run/'state_reference').mkdir()
            reference_files=['predictor_state.json','predictor_manifest.json','prime_samples.jsonl']
            if cfg['suite'].get('token_input_mode','historical') == 'predeclared_diagnostic':
                reference_files.append('planned_samples.jsonl')
            for name in reference_files:
                shutil.copy2(prime/name,run/'state_reference'/name)
        try:
            pf=preflight(cfg,run)
            if ref and read(ref/'slo_lock.json').get('environment_signature') != environment_signature(pf):
                raise Failure('environment changed since SLO calibration')
            wifi_before=jetson_wifi_health(cfg)
            save(run/'wifi_health_before.json',wifi_before)
            execute(cfg,run,stage,pf)
            wifi_after=jetson_wifi_health(cfg)
            save(run/'wifi_health_after.json',wifi_after)
            validate_jetson_wifi_health(wifi_before,wifi_after)
            audit_evidence.verify_sequence_cap(run,cfg,stage)
            if stage=='prime':
                record_prime_samples(run)
                if cfg['suite'].get('token_input_mode','historical') == 'predeclared_diagnostic':
                    audit_evidence.build_planned_feature_manifest(run,cfg)
                source=run/'server/output/predictor_state.json'
                if not source.exists(): raise Failure('server did not export the two-round predictor state')
                shutil.copy2(source,run/'predictor_state.json')
                predictor_state.read(run/'predictor_state.json')
                predictor_manifest={
                    'schema':1,'test_fingerprint':fingerprint(cfg),
                    'source_hashes':read(run/'source_hashes.json'),
                    'environment_signature':environment_signature(pf),
                    'state_sha256':digest(run/'predictor_state.json')}
                if cfg['suite'].get('token_input_mode','historical') == 'predeclared_diagnostic':
                    predictor_manifest['planned_samples_sha256']=digest(run/'planned_samples.jsonl')
                save(run/'predictor_manifest.json',predictor_manifest)
                if cfg['suite'].get('slo_source')=='prime':
                    args=analysis_args(cfg,run,stage)
                    args.out=str(run/'slo_lock.json')
                    analyze.lock_slo(args)
                    lock=read(args.out)
                    lock.update(slo_source='prime',test_fingerprint=fingerprint(cfg),
                                source_hash=digest(run/'source.tar.gz'),
                                environment_signature=environment_signature(pf),
                                state_sha256=digest(run/'predictor_state.json'),
                                calibration_round_numbers=[1,2],
                                workload_config=read(run/'effective_config.json'),
                                sample_protocol=read(run/'sample_protocol.json'),
                                tta_suite_sha256=digest(run/'suite.toml'),
                                calibration_raw_csv_sha256=digest(run/'fedavg.csv'),
                                power_trace_sha256={h['device']:digest(run/name/'power_aligned.csv')
                                                    for name,h in cfg['hosts'].items()})
                    if (run/'planned_input_features.json').is_file():
                        lock['planned_input_features']=read(run/'planned_input_features.json')
                    save(args.out,lock)
            elif stage=='calibration':
                dataset_audit=audit_evidence.build_dataset_audit(
                    run,'calibration',range(1,6),cfg['hosts'])
                if not dataset_audit['sample_nonoverlap_proven']:
                    raise Failure('prime and SLO calibration sample separation not demonstrated: '
                                  +str(dataset_audit['status']))
                planned=(read(run/'planned_input_features.json')
                         if (run/'planned_input_features.json').is_file() else None)
                # The planned manifest describes the disjoint official window.
                # Calibration trains on its own window and must not be compared
                # with the official samples' pre-training features.
                args=analysis_args(cfg,run,stage); args.out=str(run/'slo_lock.json'); analyze.lock_slo(args)
                lock=read(args.out); lock.update(test_fingerprint=fingerprint(cfg),source_hash=digest(run/'source.tar.gz'),environment_signature=environment_signature(pf),state_sha256=digest(run/'state_reference/predictor_state.json'))
                lock.update(calibration_round_numbers=list(range(1,cfg['profiles']['calibration']['num-server-rounds']+1)),
                            workload_config=read(run/'effective_config.json'),
                            sample_protocol=read(run/'sample_protocol.json'),
                            tta_suite_sha256=digest(run/'suite.toml'),
                            calibration_raw_csv_sha256=digest(run/'fedavg.csv'),
                            power_trace_sha256={h['device']:digest(run/name/'power_aligned.csv') for name,h in cfg['hosts'].items()})
                if planned:
                    lock['planned_input_features']=planned
                save(args.out,lock)
            elif stage=='official':
                lock=read(run/'reference/slo_lock.json')
                if read(run/'source_hashes.json')!=read(ref/'source_hashes.json'): raise Failure('source changed since calibration')
                if (lock.get('sample_protocol')!=read(ref/'sample_protocol.json')
                        or not calibration_official_protocol_valid(
                            lock.get('sample_protocol'),read(run/'sample_protocol.json'))):
                    raise Failure('sample selection protocol changed since SLO calibration')
                audit_evidence.build_prediction_audit(run,cfg,pf,stage)
                slo_audit(run,ref,slo_at_start)
                args=analysis_args(cfg,run,stage); args.slo=str(run/'reference/slo_lock.json'); args.outdir=str(run/'report')
                rc=analyze.score(args)
                state=read(run/'state.json'); state.update(status='complete',test_pass=rc==0); save(run/'state.json',state)
                elapsed=time.time()-entry_time
                timing={'elapsed_s':elapsed,'includes_preflight':True,
                        'target_s':cfg['suite']['onsite_target_s'],
                        'within_target':elapsed<=cfg['suite']['onsite_target_s']}
                save(run/'wall_time.json',timing)
                summary=read(run/'report/tta_summary.json')
                summary['onsite_runtime']=timing
                save(run/'report/tta_summary.json',summary)
                with (run/'report/tta_report.txt').open('a') as report:
                    report.write(f"\nOnsite elapsed (incl. preflight): {elapsed:.1f}s; "
                                 f"10-minute target: {'MET' if timing['within_target'] else 'MISSED'}\n")
                render_report.export(run)
                package(run); return rc,run
            save(run/'wall_time.json',{'elapsed_s':time.time()-entry_time,'includes_preflight':True})
            render_report.export(run)
            package(run); return 0,run
        except (Exception,KeyboardInterrupt) as e:
            if (run/'wifi_health_before.json').exists() and not (run/'wifi_health_after.json').exists():
                try: save(run/'wifi_health_after.json',jetson_wifi_health(cfg))
                except Exception as health_error: save(run/'wifi_health_after.json',{'error':str(health_error)})
            state=read(run/'state.json') if (run/'state.json').exists() else {}
            state.update(status='invalid',error=str(e)); save(run/'state.json',state)
            logs='\n'.join(p.read_text(errors='replace')[-12000:] for p in run.rglob('*.log'))
            save(run/'diagnosis.json',{'error':str(e),'suggestions':diagnosis(str(e),logs)})
            try: render_report.export(run)
            except Exception: pass
            package(run); print(f'INVALID: {e}',flush=True)
            if isinstance(e,KeyboardInterrupt) or attempt==retries or state.get('cleanup_errors'): return 2,run
            parent=run
    return 2,run

def main():
    p=argparse.ArgumentParser(); p.add_argument('-c','--config',default=str(HERE/'suite.toml'))
    subs=p.add_subparsers(dest='command',required=True)
    for command in ['preflight','smoke','prime','estimate-runtime']: subs.add_parser(command)
    s=subs.add_parser('calibrate-slo'); s.add_argument('--state',required=True)
    s=subs.add_parser('onsite'); s.add_argument('--reference',required=True)
    for command in ['resume','collect','diagnose','package','reanalyze']:
        s=subs.add_parser(command); s.add_argument('--run-dir',required=True)
    a=p.parse_args(); cfg=tomllib.loads(Path(a.config).read_text())
    try:
        validate_reuse_config(cfg)
        if a.command=='estimate-runtime':
            print(json.dumps({'estimate_s':None,'verified':False,
                              'basis':'three-round state reuse needs its own complete hardware trial'})); return 0
        if a.command=='preflight':
            run=new_run(a.config,'preflight')
            try: print(json.dumps(preflight(cfg,run),indent=2))
            finally: package(run)
            return 0
        if a.command in ['smoke','prime','calibrate-slo','onsite']:
            stage={'smoke':'smoke','prime':'prime','calibrate-slo':'calibration','onsite':'official'}[a.command]
            return trial(cfg,a.config,stage,getattr(a,'reference',None),state_reference=getattr(a,'state',None))[0]
        run=Path(a.run_dir).resolve()
        if a.command=='package': print(package(run)); return 0
        if a.command=='diagnose': print(json.dumps(read(run/'diagnosis.json'),indent=2)); return 0
        if a.command=='collect':
            state=read(run/'state.json')
            for name,root in state.get('remote_dirs',{}).items(): collect(all_hosts(cfg)[name],root,run/name)
            print(package(run)); return 0
        if a.command=='reanalyze':
            intent=read(run/'intent.json')
            if intent['stage']!='official': raise Failure('reanalyze requires an official attempt')
            if intent['fingerprint']!=fingerprint(cfg):
                raise Failure('reanalyze requires the original frozen configuration and harness')
            if (run/'report/tta_summary.json').exists(): raise Failure('completed report is immutable; use its existing result')
            state=read(run/'state.json')
            if not (run/'post_clock.json').is_file() or state.get('cleanup_errors'):
                raise Failure('reanalyze requires completed execution, valid clocks and confirmed cleanup')
            flwr=read(run/'server/flwr.state.json')
            if flwr.get('status')!='success':
                raise Failure('reanalyze requires a successfully completed Flower run')
            ref=verify_reference(cfg,intent['reference'])
            pf=read(run/'preflight.json')
            post_clock=read(run/'post_clock.json')
            audit_relative_clock(pf,post_clock,cfg['hosts'],cfg['suite']['max_clock_offset_ms']/1000)
            save(run/'post_clock.json',post_clock)
            if (read(ref/'slo_lock.json').get('sample_protocol')!=read(ref/'sample_protocol.json')
                    or not calibration_official_protocol_valid(
                        read(ref/'slo_lock.json').get('sample_protocol'),read(run/'sample_protocol.json'))):
                raise Failure('reanalyze sample selection protocol differs from SLO calibration')
            if read(run/'source_hashes.json')!=read(ref/'source_hashes.json'):
                raise Failure('reanalyze source differs from SLO calibration')
            if not (run/'fedavg.csv').exists(): normalize(cfg,run,'official',pf)
            audit_evidence.build_prediction_audit(run,cfg,pf,'official')
            start=read(run/'slo_start.json')['slo_lock_sha256']
            slo_audit(run,ref,start)
            args=analysis_args(cfg,run,'official'); args.slo=str(run/'reference/slo_lock.json'); args.outdir=str(run/'report')
            rc=analyze.score(args)
            state=read(run/'state.json'); state.update(status='complete',test_pass=rc==0); save(run/'state.json',state)
            render_report.export(run)
            package(run); return rc
        if a.command=='resume':
            intent=read(run/'intent.json')
            if intent['fingerprint']!=fingerprint(cfg): raise Failure('Resume requires the same frozen configuration and harness')
            if (run/'state.json').exists() and read(run/'state.json').get('remote_dirs'):
                if cleanup(cfg,run): raise Failure('cannot resume until owned process cleanup is confirmed')
            # Always a fresh training attempt; never splice rounds or recalibrate an onsite SLO.
            return trial(cfg,a.config,intent['stage'],intent['reference'],run,intent.get('state_reference'))[0]
    except (Exception,KeyboardInterrupt) as e:
        print('ERROR: '+str(e),file=sys.stderr); return 2
if __name__=='__main__': sys.exit(main())
