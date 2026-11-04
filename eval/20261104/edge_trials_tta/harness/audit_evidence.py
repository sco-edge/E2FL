"""TTA evidence derived from frozen run logs; never invent missing observations."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import analyze as analyze


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _one(index, key):
    values = index.get(key, [])
    return values[0] if len(values) == 1 else None


def _index(rows, fields):
    result = {}
    for row in rows:
        result.setdefault(tuple(str(row[field]) for field in fields), []).append(row)
    return result


def _float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _sample_key(row):
    digest = row.get('token_ids_sha256')
    return 'sha256:' + digest if digest else None


def verify_sequence_cap(run, cfg, stage):
    """Reject a run if any actual training input exceeds its frozen token cap."""
    run=Path(run)
    protocol=json.loads((run/'sample_protocol.json').read_text())
    if not protocol.get('sequence_length_cap_enforced'):
        return {}
    cap=int(protocol['max_seq_len'])
    if cap < 1 or cap != int(cfg['profile_common']['max-seq-len']):
        raise ValueError('TTA sequence cap differs from the frozen test configuration')
    training=_index((x for x in analyze.read_fedavg(run/'fedavg.csv')
                     if x.get('phase')=='train'),('round','device_name'))
    maxima={}
    for host,info in cfg['hosts'].items():
        device=info['device']
        observed=read_jsonl(run/host/'output/sample_audit.jsonl')
        for round_number in range(1,int(cfg['profiles'][stage]['num-server-rounds'])+1):
            rows=[x for x in observed if int(x.get('round',-1))==round_number]
            train=_one(training,(str(round_number),device))
            if train is None or not rows:
                raise ValueError(f'{device} round {round_number}: missing training or sample audit')
            lengths=[x.get('input_token_count') for x in rows]
            if any(not isinstance(length,int) or length < 1 or length > cap
                   for length in lengths):
                raise ValueError(f'{device} round {round_number}: sample exceeds {cap}-token cap')
            expected_reads=_float(train.get('num_examples'))
            epochs=_float(train.get('local_epochs'))
            actual_tokens=_float(train.get('input_tokens'))
            if (expected_reads is None or epochs is None or actual_tokens is None
                    or len(rows)!=int(expected_reads)
                    or actual_tokens!=sum(lengths)*epochs):
                raise ValueError(f'{device} round {round_number}: sample audit disagrees with training tokens')
            maxima[(round_number,device)]=max(lengths)
    return maxima


def build_planned_feature_manifest(prime_dir, cfg):
    """Use only pre-training tokenized sample features from the prime run."""
    prime_dir = Path(prime_dir)
    source = prime_dir / 'planned_samples.jsonl'
    rows = read_jsonl(source)
    cap = int(cfg['profile_common']['rpi-normal-quota'])
    expected = set(range(2 * cap, 3 * cap))
    devices = {}
    first_recorded = []
    for host in cfg['hosts'].values():
        dev = host['device']
        chosen = [x for x in rows if x.get('device_name') == dev]
        by_row = {}
        for x in chosen:
            by_row.setdefault(x.get('partition_train_row'),[]).append(x)
        if set(by_row) != expected:
            raise ValueError(f'{dev}: incomplete planned sample features')
        samples = []
        for row_number in sorted(expected):
            copies = by_row[row_number]
            x = min(copies,key=lambda item:float(item.get('recorded_at',float('inf'))))
            length = x.get('input_token_count')
            digest = x.get('token_ids_sha256')
            if (x.get('partition_id') != host['partition']
                    or x.get('model') != cfg['profile_common']['model']
                    or x.get('dataset') != cfg['profile_common']['dataset']
                    or not isinstance(length,int) or length <= 0
                    or not isinstance(digest,str) or len(digest) != 64
                    or _float(x.get('recorded_at')) is None):
                raise ValueError(f'{dev}: invalid pre-training sample feature')
            if any((other.get('partition_id'),other.get('model'),other.get('dataset'),
                    other.get('input_token_count'),other.get('token_ids_sha256'),
                    other.get('device_uid')) !=
                   (x['partition_id'],x['model'],x['dataset'],length,digest,x['device_uid'])
                   for other in copies):
                raise ValueError(f'{dev}: conflicting planned sample features')
            first_recorded.append(float(x['recorded_at']))
            samples.append({'partition_train_row':row_number,
                            'input_token_count':length,'token_ids_sha256':digest})
        devices[dev] = {'partition_id':host['partition'], 'samples':samples,
                        'input_tokens_per_epoch':sum(x['input_token_count'] for x in samples),
                        'tokens_per_example':sum(x['input_token_count'] for x in samples)/cap}
    if len({x.get('device_name') for x in rows}) != len(devices):
        raise ValueError('unexpected device in planned sample features')
    prime = read_jsonl(prime_dir / 'prime_samples.jsonl')
    if not prime or set(x['token_ids_sha256'] for x in prime) & set(
            x['token_ids_sha256'] for x in rows):
        raise ValueError('planned evaluation samples overlap prime training samples')
    return {'schema':1,'source':'prime_pretraining_tokenization',
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'last_recorded_at':max(first_recorded),
            'sample_offset':2*cap,'sample_count_cap':cap,
            'model':cfg['profile_common']['model'],
            'dataset':cfg['profile_common']['dataset'],
            'devices':devices}


def planned_sample_mismatches(run, cfg, manifest, measured_rounds):
    """Compare the locked pre-training inputs with what clients actually read."""
    run = Path(run)
    errors = {}
    fedavg = {(int(x['round']),x['device_name']):x for x in analyze.read_fedavg(run/'fedavg.csv')
              if x.get('phase') == 'train'}
    for host,h in cfg['hosts'].items():
        dev = h['device']
        planned = manifest['devices'][dev]
        expected = {x['partition_train_row']:x for x in planned['samples']}
        observed = read_jsonl(run/host/'output/sample_audit.jsonl')
        for round_number in measured_rounds:
            current = [x for x in observed if int(x['round']) == round_number]
            issues = []
            if len(current) != len(expected):
                issues.append('planned_sample_count_mismatch')
            actual_by_row = {x.get('partition_train_row'):x for x in current}
            if set(actual_by_row) != set(expected) or len(actual_by_row) != len(current):
                issues.append('planned_sample_rows_mismatch')
            for row_number,planned_row in expected.items():
                x = actual_by_row.get(row_number)
                if (not x or x.get('partition_id') != planned['partition_id']
                        or x.get('token_ids_sha256') != planned_row['token_ids_sha256']
                        or x.get('input_token_count') != planned_row['input_token_count']):
                    issues.append('planned_sample_feature_mismatch')
                    break
            training = fedavg.get((round_number,dev))
            epochs = _float(training.get('local_epochs')) if training else None
            actual_tokens = _float(training.get('input_tokens')) if training else None
            if (epochs is None or actual_tokens is None or
                    not math.isclose(actual_tokens,
                                     planned['input_tokens_per_epoch']*epochs,
                                     rel_tol=0,abs_tol=0)):
                issues.append('planned_token_workload_mismatch')
            errors[(round_number,dev)] = issues
    return errors


def build_dataset_audit(run, stage, measured_rounds, hosts):
    run = Path(run)
    (run/'report').mkdir(parents=True,exist_ok=True)
    devices=[h['device'] for h in hosts.values()]
    current = [row for host in hosts for row in
               read_jsonl(run / host / 'output/sample_audit.jsonl')]
    protocol = json.loads((run/'sample_protocol.json').read_text()) if (run/'sample_protocol.json').is_file() else {}
    two_phase=protocol.get('sample_split_policy')=='two_disjoint_windows'
    if stage == 'official' and two_phase:
        phases = {
            'prime': read_jsonl(run/'reference/prime_samples.jsonl'),
            'official': current,
        }
        rounds = {'prime':(1,2), 'official':tuple(range(1,4))}
    elif stage == 'official':
        phases = {
            'prime': read_jsonl(run/'reference/state_reference/prime_samples.jsonl'),
            'slo_calibration': [row for host in hosts for row in
                                read_jsonl(run/'reference'/host/'output/sample_audit.jsonl')],
            'official': current,
        }
        rounds = {'prime':(1,2), 'slo_calibration':(1,2,3,4,5),
                  'official':tuple(range(1,4))}
    elif stage == 'calibration':
        phases = {'prime':read_jsonl(run/'state_reference/prime_samples.jsonl'),
                  'slo_calibration':current}
        rounds = {'prime':(1,2), 'slo_calibration':tuple(range(1,6))}
    else:
        phases = {'prime':[x for x in current if int(x.get('round',-1))<=2],
                  'official':[x for x in current if int(x.get('round',-1)) in measured_rounds]}
        rounds = {'prime':(1,2), 'official':tuple(measured_rounds)}
    cap = int(protocol.get('sample_count_cap',0))
    offsets = ({'prime':cap,'official':2*cap} if two_phase else
               {'prime':0,'slo_calibration':cap,'official':2*cap})
    if stage not in ('official','calibration'):
        offsets['official']=int(protocol.get('sample_offset',0))
    partitions={h['device']:int(h['partition']) for h in hosts.values() if 'partition' in h}
    partition_valid=(len(partitions)==len(devices)
                     and len(set(partitions.values()))==len(devices)
                     and all(x.get('partition_id')==partitions.get(x.get('device_name'))
                             for rows in phases.values() for x in rows))
    phase_issues={}
    phase_ids={phase:sorted({key for row in rows if (key:=_sample_key(row))})
               for phase,rows in phases.items()}
    csv_rows=[]
    for phase,rows in phases.items():
        issues=[]
        expected_rows=set(range(offsets[phase],offsets[phase]+cap))
        if cap < 1 or not rows:
            issues.append('missing_phase_samples')
        if ((stage=='official' and phase=='official')
                or (stage=='calibration' and phase=='slo_calibration')):
            if int(protocol.get('sample_offset',-1)) != offsets[phase]:
                issues.append('run_sample_offset_mismatch')
        expected_policy='two_disjoint_windows' if two_phase else 'three_disjoint_windows'
        if stage in ('official','calibration') and protocol.get('sample_split_policy') != expected_policy:
            issues.append('missing_phase_split_policy')
        if any(x.get('device_name') not in devices
               or not isinstance(x.get('round'),int)
               or not isinstance(x.get('partition_train_row'),int)
               or not isinstance(x.get('token_ids_sha256'),str)
               or len(x['token_ids_sha256'])!=64
               or not isinstance(x.get('input_token_count'),int)
               or x['input_token_count']<1 for x in rows):
            issues.append('invalid_sample_record')
        for device in devices:
            device_rows=[x for x in rows if x.get('device_name')==device]
            if {x.get('round') for x in device_rows} != set(rounds[phase]):
                issues.append(device+':missing_or_unexpected_round')
            for round_number in rounds[phase]:
                observed=[x for x in device_rows if x.get('round')==round_number]
                if {x.get('partition_train_row') for x in observed}!=expected_rows:
                    issues.append(f'{device}:round_{round_number}:sample_window_mismatch')
            for row_number in expected_rows:
                copies=[x for x in device_rows if x.get('partition_train_row')==row_number]
                if len({(x.get('token_ids_sha256'),x.get('input_token_count')) for x in copies})!=1:
                    issues.append(f'{device}:row_{row_number}:unstable_or_missing_sample')
                if copies:
                    item=copies[0]
                    csv_rows.append({'phase':phase,'device_name':device,
                        'partition_id':item.get('partition_id'),
                        'partition_train_row':row_number,
                        'token_ids_sha256':item.get('token_ids_sha256'),
                        'input_token_count':item.get('input_token_count'),
                        'observed_rounds':','.join(str(x) for x in sorted({c.get('round') for c in copies}))})
        phase_issues[phase]=sorted(set(issues))
    names=list(phases)
    pairwise={}
    for index,left in enumerate(names):
        for right in names[index+1:]:
            pairwise[left+'_vs_'+right]=sorted(set(phase_ids[left]) & set(phase_ids[right]))
    overlap=sorted(set().union(*(set(x) for x in pairwise.values())))
    complete=all(not x for x in phase_issues.values())
    if stage=='official' and tuple(measured_rounds)!=(1,2,3):
        complete=False
    status = ('invalid_partition' if not partition_valid else
              'overlap_confirmed' if overlap else
              'disjoint_observed' if complete else 'not_demonstrated')
    result = {
        'schema': 2, 'calibration_phase': ('estimator prime also sets the SLO' if two_phase else
                                         'estimator prime and SLO calibration are separate executions'),
        'sample_id_definition': 'SHA-256 of actual fetched token ID sequence; partition-local row is retained in raw sample_audit.jsonl',
        'prime_sample_ids': phase_ids['prime'],
        'calibration_sample_ids': phase_ids['prime'] if two_phase else phase_ids.get('slo_calibration',[]),
        'slo_calibration_sample_ids': phase_ids['prime'] if two_phase else phase_ids.get('slo_calibration',[]),
        'evaluation_sample_ids': phase_ids.get('official',[]),
        'phase_sample_ids':phase_ids,
        'pairwise_overlap_sample_ids':pairwise,
        'pairwise_overlap_counts':{key:len(value) for key,value in pairwise.items()},
        'phase_issues':phase_issues,
        'overlap_sample_ids': overlap, 'overlap_count': len(overlap),
        'partition_assignment_valid':partition_valid,
        'sample_window_valid':complete,
        'sample_count_cap':cap,'phase_sample_offsets':offsets,
        'evaluation_sample_offset':offsets.get('official'),
        'observed_partitions':{d:sorted({x.get('partition_id') for rows in phases.values() for x in rows if x.get('device_name')==d}) for d in devices},
        'observed_input_token_counts':{
            d:{phase:sorted({x.get('input_token_count') for x in rows if x.get('device_name')==d and x.get('input_token_count') is not None})
               for phase,rows in phases.items()}
            for d in devices},
        'status': status, 'sample_nonoverlap_proven': status == 'disjoint_observed',
        'limitations': ('The isolated test client uses fixed disjoint phase sample windows; '
                        'it reuses each phase\'s window across rounds. Overlapping token hashes prove reuse, '
                        'but absence of overlap does not prove distinct source records if tokenization or partitioning changed. '
                        'Missing read logs never imply disjointness.'),
    }
    # Preserve the dashboard's evaluation label while exposing every phase.
    for counts in result['observed_input_token_counts'].values():
        counts['evaluation']=counts.get('official',[])
        counts['calibration']=counts.get('slo_calibration',[])
    analyze.dump_csv(run/'report/tta_dataset_samples.csv',csv_rows)
    write_json(run / 'report/tta_dataset_independence.json', result)
    return result


def build_prediction_audit(run, cfg, pf, stage):
    run = Path(run)
    report = run / 'report'; report.mkdir(exist_ok=True)
    effective_config=json.loads((run/'effective_config.json').read_text())
    protocol=json.loads((run/'sample_protocol.json').read_text())
    capped_samples=verify_sequence_cap(run,cfg,stage)
    rows = [r for r in analyze.read_fedavg(run/'fedavg.csv') if r.get('phase') == 'train']
    predictions = _index(read_jsonl(run/'server/output/predictions.jsonl'), ('round','cid'))
    post_clock=json.loads((run/'post_clock.json').read_text()) if (run/'post_clock.json').is_file() else {}
    events = []
    for host,h in cfg['hosts'].items():
        server_clock=pf['hosts']['server']['clock']; client_clock=pf['hosts'][host]['clock']
        offset = server_clock['offset_s'] - client_clock['offset_s']
        drift=_float(post_clock.get(host,{}).get('relative_drift_bound_s'))
        bound = (abs(offset) + server_clock['uncertainty_s'] + client_clock['uncertainty_s']
                 + (drift if drift is not None else 0.0))
        for e in read_jsonl(run/host/'output/training_events.jsonl'):
            e['aligned_unix_s'] = e['unix_time_s'] + offset
            e['clock_bound_s'] = bound
            e['post_clock_drift_checked'] = drift is not None
            events.append(e)
    event_index = _index(events, ('round','device_name','device_uid','kind'))
    warmup = 2 if stage=='smoke' else cfg['suite']['warmup_rounds']
    measured = set(range(warmup+1,
                         cfg['profiles'][stage]['num-server-rounds']+1))
    planned_path = run/'planned_input_features.json'
    planned = json.loads(planned_path.read_text()) if planned_path.is_file() else None
    planned_mismatches = (planned_sample_mismatches(run,cfg,planned,measured)
                          if planned else {})
    planned_source_valid = (not planned or
        planned['source_sha256'] == hashlib.sha256(
            (run/('reference' if stage=='official' and protocol.get('sample_split_policy')=='two_disjoint_windows'
                  else 'reference/state_reference' if stage=='official' else 'state_reference')/
             'planned_samples.jsonl').read_bytes()).hexdigest())
    output_inputs=[]; output_audit=[]
    for row in rows:
        r=int(row['round']); dev=row['device_name']; uid=str(row.get('device_uid',''))
        if r not in measured: continue
        p=_one(predictions,(str(r),str(row['cid'])))
        starts=_one(event_index,(str(r),dev,uid,'start'))
        ends=_one(event_index,(str(r),dev,uid,'end'))
        reasons=[]
        if not p: reasons.append('missing_or_duplicate_predispatch_record')
        if not starts or not ends: reasons.append('missing_or_duplicate_training_timestamp')
        if stage=='official' and starts and not starts['post_clock_drift_checked']:
            reasons.append('missing_post_clock_drift_audit')
        state = (p.get('model_states',{}).get(dev) if p else None)
        if not state: reasons.append('missing_predictor_state')
        if planned:
            reasons += planned_mismatches.get((r,dev),['missing_planned_sample_audit'])
            if not planned_source_valid:
                reasons.append('planned_feature_source_hash_mismatch')
            if p and float(p['predicted_at']) <= planned['last_recorded_at']:
                reasons.append('planned_features_not_frozen_before_prediction')
            planned_dev = planned['devices'].get(dev)
            if not planned_dev:
                reasons.append('missing_planned_device_features')
            elif state and not math.isclose(
                    float(state.get('tokens_per_example',0)),
                    planned_dev['tokens_per_example'],rel_tol=0,abs_tol=1e-9):
                reasons.append('planned_tokens_not_used_by_predictor')
        pred = (_float(p.get('pred_train_time_s') if p.get('device_name') else p.get('candidates',{}).get(dev))
                if p else None)
        actual = _float(row.get('actual_train_time_self'))
        if pred is None or actual is None or pred <= 0 or actual <= 0:
            reasons.append('invalid_prediction_or_actual')
        if p:
            history=p.get('history_rounds',[])
            if any(int(x)>=r for x in history): reasons.append('current_or_future_round_in_history')
            if p.get('coefficient_last_updated_round') != max(history,default=None):
                reasons.append('coefficient_history_mismatch')
            if pred is not None and _float(row.get('pred_train_time_self')) is not None and not math.isclose(pred,float(row['pred_train_time_self']),rel_tol=1e-9):
                reasons.append('normalized_prediction_mismatch')
        if state and pred is not None:
            beta=_float(state.get('beta_sec_per_token')); alpha=_float(state.get('alpha'))
            work=_float(state.get('token_workload'))
            if None in (beta,alpha,work) or not math.isclose(pred,beta*alpha*work,rel_tol=1e-6,abs_tol=1e-6):
                reasons.append('predictor_formula_mismatch')
            if _float(row.get('max_samples')) != _float(state.get('effective_samples')):
                reasons.append('delivered_sample_cap_mismatch')
            if _float(row.get('local_epochs')) != _float(state.get('effective_epochs')):
                reasons.append('delivered_epochs_mismatch')
        workload=starts.get('workload') if starts else None
        config=p.get('config',{}) if p else {}
        if not isinstance(workload,dict):
            reasons.append('missing_client_workload_record')
        elif state:
            expected={
                'max_samples_per_round':state.get('effective_samples'),
                'selected_samples':row.get('max_samples'),
                'local_steps':state.get('effective_steps'),
                'local_epochs':state.get('effective_epochs'),
                'batch_size':config.get('batch_size'),
                'max_seq_len_configured':config.get('max_seq_len'),
                'lora_rank':row.get('lora_rank'),
            }
            for field,value in expected.items():
                if _float(workload.get(field)) != _float(value):
                    reasons.append('client_'+field+'_mismatch')
            if _float(workload.get('sample_offset')) != _float(protocol.get('sample_offset')):
                reasons.append('client_sample_offset_mismatch')
            expected_modules=sorted(x.strip() for x in str(effective_config.get('lora-target-modules','')).split(',') if x.strip())
            active_modules=workload.get('active_lora_modules')
            if not isinstance(active_modules,list) or sorted(active_modules)!=expected_modules:
                reasons.append('client_active_lora_modules_mismatch')
            if workload.get('model')!=effective_config.get('model'):
                reasons.append('client_model_mismatch')
            if workload.get('dataset')!=effective_config.get('dataset'):
                reasons.append('client_dataset_mismatch')
        if p and starts:
            if float(p['predicted_at']) + starts['clock_bound_s'] >= starts['aligned_unix_s']:
                reasons.append('prediction_not_proven_before_training')
        if starts and ends:
            if starts['aligned_unix_s'] >= ends['aligned_unix_s']:
                reasons.append('training_timestamp_order_invalid')
            if actual is not None and abs(ends['aligned_unix_s']-starts['aligned_unix_s']-actual)>1.0:
                reasons.append('training_duration_mismatch')
        input_row={
            'round':r,'logical_round':r+2 if stage in ('official','calibration') else r,
            'device_name':dev,'device_uid':uid,
            'max_samples_per_round':state.get('effective_samples') if state else None,
            'client_max_samples_per_round':workload.get('max_samples_per_round') if isinstance(workload,dict) else None,
            'selected_samples':workload.get('selected_samples') if isinstance(workload,dict) else None,
            'local_steps':state.get('effective_steps') if state else None,
            'executed_local_steps':row.get('local_steps'),
            'batch_size':config.get('batch_size'), 'max_seq_len':config.get('max_seq_len'),
            'sequence_length_cap_enforced':bool(protocol.get('sequence_length_cap_enforced')),
            'max_observed_input_tokens':capped_samples.get((r,dev)),
            'local_epochs':state.get('effective_epochs') if state else None,
            'lora_rank':row.get('lora_rank') or effective_config.get('lora-r'),
            'lora_modules':','.join(workload['active_lora_modules']) if isinstance(workload,dict) and isinstance(workload.get('active_lora_modules'),list) else None,
            'lora_modules_declared':effective_config.get('lora-target-modules'),
            'lora_modules_flower_config':workload.get('lora_modules_configured') if isinstance(workload,dict) else None,
            'beta_sec_per_token':state.get('beta_sec_per_token') if state else None,
            'alpha':state.get('alpha') if state else None,
            'estimated_tokens_per_example':state.get('tokens_per_example') if state else None,
            'token_workload':state.get('token_workload') if state else None,
            'token_workload_source':('frozen_pretraining_sample_features' if planned else 'historical_mean'),
            'planned_input_tokens_per_epoch':(planned['devices'][dev]['input_tokens_per_epoch']
                                               if planned and dev in planned['devices'] else None),
            'planned_feature_source_sha256':planned.get('source_sha256') if planned else None,
            'coefficient_last_updated_round':p.get('coefficient_last_updated_round') if p else None,
            'prediction_timestamp':p.get('predicted_at') if p else None,
            'pred_train_time_self':pred,'actual_train_time_self':actual,
            'model':workload.get('model') if isinstance(workload,dict) else None,
            'dataset':workload.get('dataset') if isinstance(workload,dict) else None,
            'predictor_inputs':('device identity; effective sample cap; epochs; local steps when >0; '
                                + ('predeclared tokenized sample lengths' if planned else 'observed tokens/example')
                                + '; beta; alpha'),
            'conditional_predictor_inputs':'batch size when step cap >0; sequence cap only in history-free fallback',
            'training_environment_only':'LoRA rank; LoRA modules; model; dataset',
        }
        output_inputs.append(input_row)
        output_audit.append({
            'round':r,'logical_round':r+2 if stage in ('official','calibration') else r,
            'device_name':dev,'device_uid':uid,
            'prediction_timestamp':p.get('predicted_at') if p else None,
            'training_start_timestamp':starts['aligned_unix_s'] if starts else None,
            'training_end_timestamp':ends['aligned_unix_s'] if ends else None,
            'clock_alignment_bound_s':starts['clock_bound_s'] if starts else None,
            'coefficient_last_updated_round':p.get('coefficient_last_updated_round') if p else None,
            'history_rounds':json.dumps(p.get('history_rounds',[])) if p else None,
            'priming_source_rounds':json.dumps([1,2]) if stage in ('official','calibration') else None,
            'token_workload_source':('frozen_pretraining_sample_features' if planned else 'historical_mean'),
            'planned_feature_source_sha256':planned.get('source_sha256') if planned else None,
            'pred_train_time_self':pred,'actual_train_time_self':actual,
            'absolute_error_s':abs(pred-actual) if pred is not None and actual is not None else None,
            'relative_error':abs(pred-actual)/actual if pred is not None and actual is not None and actual>0 else None,
            'valid_prediction':not reasons,'invalid_reasons':';'.join(reasons),
        })
    analyze.dump_csv(report/'tta_prediction_inputs.csv',output_inputs)
    analyze.dump_csv(report/'tta_prediction_audit.csv',output_audit)
    dataset_audit=build_dataset_audit(run,stage,sorted(measured),cfg['hosts'])
    dataset_audit['runtime_observation_independence'] = bool(output_audit) and all(
        'current_or_future_round_in_history' not in x['invalid_reasons']
        and 'coefficient_history_mismatch' not in x['invalid_reasons']
        and x['prediction_timestamp'] is not None
        and x['training_start_timestamp'] is not None
        and x['prediction_timestamp'] + x['clock_alignment_bound_s'] < x['training_start_timestamp']
        for x in output_audit)
    write_json(report/'tta_dataset_independence.json',dataset_audit)
    return output_audit


def build_energy_audit(run, cfg, observations):
    run=Path(run)
    boundary=_index(read_jsonl(run/'server/output/round_boundaries.jsonl'),('round',))
    predictions=read_jsonl(run/'server/output/predictions.jsonl')
    preflight=json.loads((run/'preflight.json').read_text())
    post_clock=json.loads((run/'post_clock.json').read_text())
    host_for_device={'RPi5':'rpi5','jetson_orin_nx':'jetson'}
    rows=[]
    for o in observations:
        round_no=int(o['round'])
        b=_one(boundary,(str(round_no),))
        delta=abs(float(o['round_end_unix_s'])-float(b['ended_at'])) if b else None
        if round_no==1:
            first=[_float(x.get('predicted_at')) for x in predictions if int(x.get('round',-1))==1]
            first=[x for x in first if x is not None]
            start_event=min(first) if first else None
        else:
            previous=_one(boundary,(str(round_no-1),))
            start_event=_float(previous.get('ended_at')) if previous else None
        start_delta=abs(float(o['round_start_unix_s'])-start_event) if start_event is not None else None
        host=host_for_device[o['device_name']]
        clock_bound=max(preflight['hosts'][host]['server_alignment_bound_s'],
                        post_clock[host]['server_alignment_bound_s'],
                        post_clock[host]['relative_drift_bound_s'])
        clock_valid=clock_bound <= cfg['suite']['max_clock_offset_ms']/1000
        valid=(b is not None and delta <= 2.0 and start_delta is not None and start_delta <= 2.0
               and float(o['max_power_gap_s']) <= cfg['suite']['max_power_sample_gap_s']
               and clock_valid)
        rows.append({
            'round':o['round'],'logical_round':int(o['round'])+2,
            'device_name':o['device_name'],
            'round_start_timestamp':o['round_start_unix_s'],
            'round_end_timestamp':o['round_end_unix_s'],
            'round_duration_s':o['round_duration_s'],
            'power_samples':o['power_samples'],
            'max_power_sample_gap_s':o['max_power_gap_s'],
            'round_energy_j':o['round_energy_j'],
            'energy_slo_j':o['energy_slo_j'],
            'energy_violation':o['energy_violation'],
            'boundary_source':'server fedavg evaluate timestamp; previous evaluate for next start',
            'start_boundary_delta_s':start_delta,
            'evaluate_completion_delta_s':delta,
            'clock_alignment_bound_s':clock_bound,
            'clock_valid':clock_valid,
            'valid_energy':valid,
        })
    analyze.dump_csv(run/'report/tta_energy_audit.csv',rows)
    return rows
