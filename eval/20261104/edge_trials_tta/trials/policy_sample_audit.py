"""Export observed sample IDs without modifying signed experiment evidence."""
import csv
import hashlib
import json
from pathlib import Path
import re

from run_trial import integrity


def export(run, output):
    run=Path(run).resolve();output=Path(output).resolve()
    if output.is_relative_to(run):raise ValueError('derived audit must be outside signed run')
    good,reason=integrity(run)
    if not good:raise ValueError(reason)
    read=lambda p:json.loads(p.read_text())
    if read(run/'state.json').get('status')!='complete':raise ValueError('run is incomplete')
    condition=read(run/'condition.json');manifest_path=run/'partition_manifest.json'
    digest=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest!=condition['manifest_sha256']:raise ValueError('manifest hash mismatch')
    manifest=read(manifest_path)
    with (run/'policy_rounds.csv').open() as stream:
        work={(int(r['round']),r['device_name']):r for r in csv.DictReader(stream)}
    predictions={}
    warmup_predictions={}
    for path in (run/'server/output').glob('*/server/policy_predictions.jsonl'):
        for line in path.read_text().splitlines():
            record=json.loads(line)
            if int(record['round'])<=condition['warmup_rounds']:
                warmup_predictions[(int(record['round']),record['device_name'])]=record
    if (output/'prediction_check.csv').exists():
        with (output/'prediction_check.csv').open() as stream:
            predictions={(int(r['round']),r['device']):r for r in csv.DictReader(stream)}
    sample_rows=[];round_rows=[];devices={};global_sets={}
    for host,device in (('rpi5','RPi5'),('jetson','jetson_orin_nx')):
        argv=read(run/host/'supernode.job.json')['argv']
        node=argv[argv.index('--node-config')+1]
        partition=int(re.search(r'partition-id=(\d+)',node).group(1))
        mapping=manifest['clients'][str(partition)]['train_indices']
        files=list((run/host/'output').glob('*/clients/exposure_state/*/round_*.json'))
        entries={}
        for path in files:
            entry=read(path);r=int(entry['round'])
            if r in entries:raise ValueError('duplicate sample evidence round')
            entries[r]=(entry,path)
        expected={r for (r,d),row in work.items() if d==device and int(row['selected_samples'])>0}
        if set(entries)!=expected:raise ValueError('missing or unexpected sample evidence')
        previous=set();calibration=set();evaluation=set();visits=0;duplicate_visits=0
        for r,(entry,path) in sorted(entries.items()):
            ids=[int(i) for i in entry['visited_sample_ids']]
            if (len(ids)!=entry['executed_examples'] or len(ids)!=int(work[(r,device)]['selected_samples'])
                    or entry['eligible_unique_samples']!=len(mapping)):
                raise ValueError('executed sample count differs from evidence')
            if any(i<0 or i>=len(mapping) for i in ids):raise ValueError('sample ID outside partition')
            global_ids=[mapping[i] for i in ids];current=set(global_ids)
            warmup=r<=condition['warmup_rounds'];phase='warmup' if warmup else 'evaluation'
            prediction=predictions.get((r,device),{})
            warmup_record=warmup_predictions.get((r,device),{})
            predicted=prediction.get('predicted_s',warmup_record.get('predicted_s',''))
            actual=work[(r,device)].get('actual_train_time_s',prediction.get('actual_s',''))
            error=prediction.get('relative_error_pct','')
            success=prediction.get('within_10pct','') if not warmup else ''
            common={'round':r,'segment':phase,'device_name':device,
                'evaluation_status':'warmup_excluded' if warmup else ('evaluated' if success in ('True','False') else 'not_measured'),
                'predicted_s':predicted,'actual_train_time_s':actual,
                'absolute_error_s':abs(float(predicted)-float(actual)) if predicted!='' and actual!='' else '',
                'relative_error_pct':error,'prediction_success':success,
                'one_minus_relative_error_pct':100-float(error) if error!='' else '',
                **work[(r,device)],'partition_id':partition,
                'prediction_source':prediction.get('prediction_source','before_dispatch' if warmup_record else ''),
                'requested_samples':prediction.get('requested_samples',''),
                'requested_step_cap':prediction.get('requested_step_cap',''),
                'sample_source_json':str(path.relative_to(run)),'manifest_sha256':digest}
            round_rows.append({**common,'sample_global_ids':json.dumps(global_ids),
                'sample_local_ids':json.dumps(ids),'unique_samples':len(current),
                'previous_rounds_overlap_count':len(current&previous),
                'calibration_overlap_count':len(current&calibration) if not warmup else 0,
                'within_round_duplicate_visits':len(ids)-len(current)})
            seen_round=set()
            for ordinal,(local,global_id) in enumerate(zip(ids,global_ids),1):
                sample_rows.append({**common,'visit_ordinal':ordinal,'sample_local_id':local,
                    'sample_global_id':global_id,'seen_in_previous_rounds':global_id in previous,
                    'seen_in_calibration':global_id in calibration if not warmup else False,
                    'repeated_within_round':global_id in seen_round})
                duplicate_visits+=int(global_id in previous or global_id in seen_round)
                seen_round.add(global_id)
            previous.update(current)
            (calibration if warmup else evaluation).update(current)
            visits+=len(ids)
        devices[device]={'partition_id':partition,'observed_rounds':len(entries),
            'total_visits':visits,'unique_global_ids':len(previous),
            'calibration_unique_ids':len(calibration),'evaluation_unique_ids':len(evaluation),
            'calibration_evaluation_overlap':len(calibration&evaluation),
            'repeat_visits':duplicate_visits}
        global_sets[device]=previous
    output.mkdir(parents=True,exist_ok=True)
    round_rows.sort(key=lambda r:(int(r['round']),r['device_name']))
    sample_rows.sort(key=lambda r:(int(r['round']),r['device_name'],r['visit_ordinal']))
    accuracy_rows=[]
    for device in ('overall','RPi5','jetson_orin_nx'):
        selected=[r for r in round_rows if r['segment']=='evaluation' and (device=='overall' or r['device_name']==device)]
        scored=[r for r in selected if r['evaluation_status']=='evaluated']
        count=sum(r['prediction_success']=='True' for r in scored)
        accuracy_rows.append({'device':device,'successful_predictions':count,
            'evaluated_device_rounds':len(scored),'expected_device_rounds':len(selected),
            'prediction_accuracy_pct':100*count/len(scored) if scored else '',
            'mean_relative_error_pct':sum(float(r['relative_error_pct']) for r in scored)/len(scored) if scored else '',
            'one_minus_mare_pct':100-sum(float(r['relative_error_pct']) for r in scored)/len(scored) if scored else '',
            'relative_error_limit_pct':10,'target_accuracy_pct':90,
            'metric_scope':'device-round; sample-level repeated rows must not be counted separately',
            'tta_pass':'not_verified'})
    for name,data in (('sample_visits.csv',sample_rows),('sample_workload_audit.csv',round_rows),('prediction_accuracy_summary.csv',accuracy_rows)):
        with (output/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    summary={'run':str(run),'manifest_sha256':digest,'devices':devices,
        'cross_device_global_id_overlap':len(global_sets['RPi5']&global_sets['jetson_orin_nx']),
        'scope':'Observed dataset row IDs only; identical text in different dataset rows is not checked.',
        'tta_pass':None}
    (output/'sample_independence.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    names=('sample_visits.csv','sample_workload_audit.csv','sample_independence.json','prediction_accuracy_summary.csv')
    (output/'sample_audit_checksums.sha256').write_text(''.join(
        hashlib.sha256((output/name).read_bytes()).hexdigest()+'  '+name+'\n' for name in names))
    return summary
