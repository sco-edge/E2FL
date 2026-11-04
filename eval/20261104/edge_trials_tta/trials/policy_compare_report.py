#!/usr/bin/env python3
"""Read-only comparison of three signed exploratory policy attempts."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import html
import json
import math
import os
from pathlib import Path
import sys

from run_trial import integrity

HERE=Path(__file__).resolve().parent
MODES=('uniform','loracast-off','loracast-on')
OUT=HERE/'policy_comparisons'


def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))


def prediction_check(run, condition):
    """Numerical check of logged predictions; never asserts TTA audit validity."""
    files=list((run/'server/output'/condition['config']['experiment-id']/'server').glob('fedavg_*.csv'))
    if len(files)!=1:raise ValueError('exactly one original prediction CSV is required')
    audit_file=files[0].parent/'policy_predictions.jsonl'
    audits={}
    if audit_file.exists():
        for line in audit_file.read_text().splitlines():
            item=json.loads(line);key=(int(item['round']),item['device_name'])
            if key in audits:raise ValueError(f'duplicate pre-dispatch prediction: {key}')
            if any(int(r)>=key[0] for r in item['history_rounds']):
                raise ValueError('pre-dispatch forecast contains current/future history')
            audits[key]=item
    expected={(r,d) for r in range(condition['warmup_rounds']+1,condition['rounds']+1)
              for d in ('RPi5','jetson_orin_nx')}
    observed={}
    for row in rows(files[0]):
        if row.get('phase')!='train':continue
        key=(int(row['round']),row['device_name'])
        if key not in expected:continue
        if key in observed:raise ValueError(f'duplicate original prediction: {key}')
        observed[key]=row
    details=[]
    for round_number,device in sorted(expected):
        row=observed.get((round_number,device),{})
        audit=audits.get((round_number,device),{})
        prediction=audit.get('predicted_s') if audit_file.exists() else row.get('pred_train_time_self')
        try:
            predicted=float(prediction);actual=float(row['actual_train_time_self'])
            available=math.isfinite(predicted) and predicted>=0 and math.isfinite(actual) and actual>0
        except (KeyError,ValueError,TypeError):available=False
        error=abs(predicted-actual)/actual if available else None
        details.append({'mode':condition['mode'],'round':round_number,'device':device,
            'predicted_s':predicted if available else '',
            'group_prediction_logged_s':audit.get('group_predicted_s',row.get('pred_train_time_'+device,'')),
            'post_reply_predicted_s':row.get('pred_train_time_self',''),
            'prediction_source':'before_dispatch' if audit_file.exists() else 'post_reply_history_only',
            'prediction_timestamp':audit.get('predicted_at',''),
            'estimated_tokens_per_example':audit.get('estimated_tokens_per_example',''),
            'requested_samples':audit.get('requested_samples',row.get('max_samples','')),
            'requested_step_cap':audit.get('config',{}).get('local_steps_'+device,'') if audit else '',
            'actual_s':actual if available else '',
            'actual_samples':row.get('num_examples',''),
            'actual_input_tokens':row.get('input_tokens',''),
            'beta_sec_per_token':row.get('est_beta_self',''),
            'alpha':row.get('est_alpha_self',''),
            'relative_error_pct':100*error if available else '',
            'within_10pct':error<=0.1 if available else '',
            'observation_status':'logged' if available else 'not_measured',
            'source_csv':str(files[0].relative_to(run))})
    measured=[r for r in details if r['observation_status']=='logged']
    successes=sum(r['within_10pct'] for r in measured)
    accuracy=successes/len(measured) if measured else None
    complete=len(measured)==len(expected)
    summary={'mode':condition['mode'],'successful_predictions':successes,
        'observations':len(measured),'expected_observations':len(expected),
        'accuracy_pct':100*accuracy if accuracy is not None else None,
        'numerical_status':('meets_target' if accuracy>=0.9 else 'below_target') if complete and measured else 'incomplete',
        'prediction_source':'before_dispatch' if audit_file.exists() else 'post_reply_history_only',
        'prediction_audit_status':'not_verified','tta_pass':None}
    return summary,details


def single_report(run):
    """Render one completed, checksum-verified attempt without training again."""
    run=Path(run).resolve()
    good,why=integrity(run)
    if not good:raise ValueError(f'{run.name}: {why}')
    state=read(run/'state.json')
    if state.get('status')!='complete':raise ValueError('실행이 정상 완료되지 않아 결과를 확정할 수 없습니다.')
    condition=read(run/'condition.json')
    check,detail=prediction_check(run,condition)
    work=rows(run/'policy_rounds.csv')
    out=HERE/'policy_reports'/run.name
    out.mkdir(parents=True,exist_ok=True)
    for name,data in (('prediction_check.csv',detail),('policy_rounds.csv',work)):
        with (out/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
    measured=[r for r in detail if r['observation_status']=='logged']
    auxiliary=(100-sum(r['relative_error_pct'] for r in measured)/len(measured)) if measured else None
    summary={'run':str(run),'condition':condition,'prediction_check':check,
             'one_minus_mare_pct':auxiliary,'tta_pass':None,'energy_slo_status':'not_audited'}
    from policy_sample_audit import export as export_samples
    sample_audit=export_samples(run,out)
    summary['sample_independence']=sample_audit
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    esc=lambda value:html.escape(str(value),quote=True)
    def table(data,columns):
        head='<tr>'+''.join('<th>'+esc(label)+'</th>' for key,label in columns)+'</tr>'
        body=''.join('<tr>'+''.join('<td>'+esc(row.get(key,''))+'</td>' for key,label in columns)+'</tr>' for row in data)
        return '<div class="scroll"><table>'+head+body+'</table></div>'
    predictions=table(detail,[('round','Round'),('device','Device'),('requested_samples','할당 S'),
        ('requested_step_cap','할당 H'),('actual_samples','실제 샘플'),
        ('estimated_tokens_per_example','예상 tokens/sample'),('actual_input_tokens','실제 총 토큰'),
        ('prediction_source','예측 출처'),('predicted_s','개별 예측 s'),
        ('group_prediction_logged_s','그룹 예측 s'),('actual_s','실제 s'),
        ('relative_error_pct','상대오차 %'),('within_10pct','오차 ≤10%')])
    workloads=table(work,[('round','Round'),('segment','구간'),('device_name','Device'),
        ('selected_samples','실제 샘플'),('input_tokens','실제 토큰'),('local_steps','실행 steps'),
        ('actual_train_time_s','학습 s'),('tbm_action','NX TBM 동작')])
    accuracy=f"{check['accuracy_pct']:.2f}%" if check['accuracy_pct'] is not None else 'Not measured'
    aux=f'{auxiliary:.2f}%' if auxiliary is not None else 'Not measured'
    status={'below_target':'90% 기준 미달','meets_target':'수치상 충족 · 감사 미검증','incomplete':'표본 부족'}[check['numerical_status']]
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LoRACast 실행 결과</title><style>body{{font:15px/1.6 system-ui;max-width:1150px;margin:30px auto;padding:0 20px;color:#193047;background:#f6f9fc}}section{{background:white;padding:20px;margin:18px 0;border:1px solid #dbe4ee;border-radius:12px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}th{{white-space:nowrap}}.note{{color:#89420d}}</style>
<h1>LoRACast 실행 결과 · {esc(condition['mode'])}</h1>
<p>{esc(condition['model'])} · {esc(condition['dataset'])}<br>{esc(run.name)} · {condition['rounds']} rounds / 처음 {condition['warmup_rounds']}회 warm-up</p>
<p>설정: L={condition['config']['max-seq-len']}, B={condition['config']['batch_size']}, E={condition['config']['local-epochs']}, Pi S={condition['config']['rpi-normal-quota']}, Jetson S 후보 {condition['config']['jetson-base-quota']}–{condition['config']['jetson-max-quota']}. H=0은 step 상한 없음입니다. L은 최대 길이이며 실제 토큰량과 다를 수 있습니다.</p>
<section><h2>예측 수치 확인: {status}</h2><p>오차 10% 이내: <b>{check['successful_predictions']}/{check['observations']} ({accuracy})</b> · 목표 90%<br>보조 지표 1−MARE: {aux}</p>
<p class="note">TTA 통과 미확인. 예측 시점·데이터 독립성 감사 및 사전 SLO 검증이 없는 비교 실행입니다. TC4 에너지 SLO: Not measured.</p>
<p>예측 출처: {esc(check['prediction_source'])}. before_dispatch는 학습 요청 반환 전에 저장한 예측입니다. post_reply_history_only는 응답 후 현재 관측을 history에 넣기 전에 재계산한 기존 CSV 값입니다. 전체 TTA 시점 감사는 별도로 필요합니다.</p>{predictions}</section>
<section><h2>라운드별 실제 작업량</h2>{workloads}</section>
<section><h2>데이터</h2><p>이 비교는 exposure-aware 샘플 선택을 사용합니다. 기존 TTA의 고정 샘플 구간과 조건이 다릅니다.</p>
<a href="prediction_check.csv">예측 CSV</a> · <a href="policy_rounds.csv">작업량 CSV</a> · <a href="summary.json">조건·요약 JSON</a></section></html>'''
    overlap=', '.join(f"{device}: {values['calibration_evaluation_overlap']}개" for device,values in sample_audit['devices'].items())
    section=f'''<section><h2>실제 사용 샘플 검증</h2><p>보정·평가 데이터 행 ID 교집합: {esc(overlap)}.<br>기기 간 데이터 행 ID 교집합: {sample_audit['cross_device_global_id_overlap']}개.</p>
<p><a href="sample_workload_audit.csv">라운드별 샘플 ID·작업량·예측 통합 CSV</a> · <a href="prediction_accuracy_summary.csv">예측 정확도 요약 CSV</a> · <a href="sample_visits.csv">샘플별 실제 방문 CSV</a> · <a href="sample_independence.json">중복 검증 JSON</a></p>
<p>기기 내부 ID를 분할 manifest의 원본 데이터 행 ID로 변환해 비교했습니다. 다른 행에 같은 문장이 들어 있는지까지 검사한 것은 아닙니다.</p></section>'''
    page=page.replace('</body>','') if '</body>' in page else page
    page=page.replace('</html>',section+'</html>')
    (out/'index.html').write_text(page)
    return out/'index.html'


def compare(runs,out=None):
    runs=[Path(x).resolve() for x in runs]
    if len(runs)!=3:raise ValueError('exactly three run directories are required')
    by_mode={}
    for run in runs:
        good,why=integrity(run)
        if not good:raise ValueError(f'{run.name}: {why}')
        state=read(run/'state.json')
        if state.get('status')!='complete':raise ValueError(f'{run.name}: incomplete attempt')
        condition=read(run/'condition.json')
        summary=read(run/'policy_summary.json')
        mode=condition['mode']
        if mode in by_mode:raise ValueError(f'duplicate mode: {mode}')
        by_mode[mode]=(run,condition,summary,rows(run/'policy_rounds.csv'))
    if set(by_mode)!=set(MODES):raise ValueError('uniform, loracast-off and loracast-on are required')
    shared=('model','dataset','manifest_sha256','rounds','warmup_rounds','seed')
    for key in shared:
        if len({str(item[1][key]) for item in by_mode.values()})!=1:
            raise ValueError(f'comparison condition differs: {key}')
    if len({json.dumps(read(item[0]/'source_hashes.json'),sort_keys=True)
            for item in by_mode.values()})!=1:
        raise ValueError('compared attempts use different source app snapshots')
    common_keys=('rpi-normal-quota','max-seq-len','batch_size','local-epochs',
                 'sample-selection-mode','sample-seed','aggregation-mode',
                 'lora-r','lora-target-modules','use-comm-prediction')
    for key in common_keys:
        if len({str(item[1]['config'][key]) for item in by_mode.values()})!=1:
            raise ValueError(f'comparison training condition differs: {key}')
    if by_mode['loracast-off'][1]['config']['jetson-max-quota'] != by_mode['loracast-on'][1]['config']['jetson-max-quota']:
        raise ValueError('TBM off/on use different NX candidate ceilings')
    records=[];totals=[];prediction_summaries=[];prediction_rows=[]
    for mode in MODES:
        run,condition,summary,data=by_mode[mode]
        prediction_summary,prediction_detail=prediction_check(run,condition)
        prediction_summaries.append(prediction_summary);prediction_rows.extend(prediction_detail)
        expected=condition['rounds']*2
        if len(data)!=expected:raise ValueError(f'{mode}: incomplete device-round rows')
        for row in data:
            records.append({'mode':mode,'run':run.name,'round':int(row['round']),
                'segment':row['segment'],'device':row['device_name'],
                'samples':int(row['selected_samples']),'tokens':int(row['input_tokens']),
                'steps':int(row['local_steps']) if row['local_steps'] else '',
                'train_s':float(row['actual_train_time_s']) if row['actual_train_time_s'] else '',
                'aggregation_weight':row['aggregation_weight'],
                'nx_tbm_action':row['tbm_action'],
                'nx_remaining_before':row['tbm_remaining_before']})
        evaluation=[r for r in data if r['segment']=='evaluation']
        totals.append({'mode':mode,'run':run.name,
            'pi_samples':sum(int(r['selected_samples']) for r in evaluation if r['device_name']=='RPi5'),
            'nx_samples':sum(int(r['selected_samples']) for r in evaluation if r['device_name']=='jetson_orin_nx'),
            'pi_tokens':sum(int(r['input_tokens']) for r in evaluation if r['device_name']=='RPi5'),
            'nx_tokens':sum(int(r['input_tokens']) for r in evaluation if r['device_name']=='jetson_orin_nx'),
            'nx_omitted_rounds':summary['nx_omitted_rounds'],
            'mac_attempt_s':round(read(run/'state.json')['end']-read(run/'state.json')['start'],1)})
    if out is None:
        out=OUT/datetime.now().strftime('%Y%m%d_%H%M%S')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    with (out/'policy_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    with (out/'prediction_check.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(prediction_rows[0]));writer.writeheader();writer.writerows(prediction_rows)
    on_condition=by_mode['loracast-on'][1]
    on_rows=by_mode['loracast-on'][3]
    used_visits=sum(int(r['selected_samples']) for r in on_rows
                    if r['device_name']=='jetson_orin_nx')
    eligible_unique=int(on_condition['config']['nx-eligible-unique'])
    budget_visits=math.floor(float(on_condition['config']['exposure-max'])*eligible_unique+1e-9)
    tbm_note=('TBM uses the paper Exposure-2 ratio; it may not bind in this short run.'
              if on_condition.get('exposure_max')==2 else
              'The short TBM cap is a control-path demonstration, not the paper Exposure-2 setting.')
    result={'schema':1,'status':'observed_control_comparison','model':by_mode['uniform'][1]['model'],
        'dataset':by_mode['uniform'][1]['dataset'],'manifest_sha256':by_mode['uniform'][1]['manifest_sha256'],
        'rounds':by_mode['uniform'][1]['rounds'],'warmup_rounds':2,
        'seed':by_mode['uniform'][1]['seed'],'summary':totals,
        'tbm_exposure_max':on_condition.get('exposure_max'),
        'tbm_cap_samples':on_condition.get('tbm_cap_samples'),
        'tbm_budget':{'jetson_used_visits':used_visits,
                      'jetson_limit_visits':budget_visits,
                      'jetson_eligible_unique':eligible_unique},
        'tta_pass':None,'energy_slo_status':'not_audited',
        'logged_prediction_checks':prediction_summaries,
        'notes':['Same model, dataset manifest, seed, LoRA setup and sampler across modes.',
                 tbm_note,
                 f'TBM budget: {used_visits} cumulative Jetson sample visits / {budget_visits} allowed.',
                 'Actual sampled rows and token lengths can differ across modes as allocation and visit history diverge.',
                 'This comparison uses a manifest partition and exposure-aware sampling, unlike the existing TTA fixed sample-window path; it is not a policy-only change to that TTA trial.',
                 'The allocation uses device-group predictions. pred_train_time_self is recomputed for the individual device after replies, before committing current timing history; it is not a saved pre-dispatch audit.',
                 'Prediction success uses absolute(predicted - actual) / actual <= 0.10; target success rate is 90%. Prediction timing and calibration/evaluation independence are not audited in these policy attempts.',
                 'Power traces are retained, but no TTA whole-round energy audit or PASS/FAIL applies.']}
    (out/'policy_comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    esc=lambda x:html.escape(str(x),quote=True)
    header='<tr>'+''.join('<th>'+esc(x)+'</th>' for x in ('Mode','Pi samples','Jetson samples','Pi tokens','Jetson tokens','NX omitted','Attempt time'))+'</tr>'
    table=''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in
        (r['mode'],r['pi_samples'],r['nx_samples'],r['pi_tokens'],r['nx_tokens'],
         ', '.join(map(str,r['nx_omitted_rounds'])) or 'none',str(r['mac_attempt_s'])+' s'))+'</tr>' for r in totals)
    detail_header='<tr>'+''.join('<th>'+esc(x)+'</th>' for x in ('Mode','Round','Phase','Device','Samples','Tokens','Steps','Train s','Weight','NX TBM action'))+'</tr>'
    detail=''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in
        (r['mode'],r['round'],r['segment'],r['device'],r['samples'],r['tokens'],
         r['steps'],r['train_s'],r['aggregation_weight'],r['nx_tbm_action']))+'</tr>' for r in records)
    notes=''.join('<li>'+esc(n)+'</li>' for n in result['notes'])
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LoRACast policy comparison</title><style>body{{font:15px/1.5 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;max-width:1200px;margin:30px auto;padding:0 18px;color:#193047;background:#f6f9fc}}h1{{margin-bottom:4px}}.sub{{color:#51647a}}section{{background:#fff;border:1px solid #dbe4ee;border-radius:14px;padding:18px;margin:18px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:720px}}td,th{{border-bottom:1px solid #e0e8f0;padding:8px;text-align:left}}th{{background:#eef4fa}}.note{{color:#8a4914}}a{{color:#245e9e}}</style><h1>LoRACast · TBM 비교</h1><p class="sub">{esc(result['model'])} · {esc(result['dataset'])} · {result['rounds']} rounds (처음 2회 warm-up) · 실제 장비 관측</p><section><h2>평가 라운드 합계</h2><div class="scroll"><table>{header}{table}</table></div><p class="note">TTA TC1/TC4 PASS/FAIL과 에너지 절감 판정은 이 비교에 적용하지 않았습니다.</p></section><section><h2>기기·라운드 실제 작업량</h2><div class="scroll"><table>{detail_header}{detail}</table></div></section><section><h2>해석 범위</h2><ul>{notes}</ul><p><a href="policy_comparison.csv">원본 연결 CSV</a> · <a href="policy_comparison.json">조건 및 요약 JSON</a></p></section></html>'''
    labels={'below_target':'90% 기준 미달','meets_target':'수치상 충족 · 증빙 미검증','incomplete':'표본 부족 · 판정 불가'}
    prediction_table='<tr><th>방식</th><th>오차 ≤10% 표본</th><th>예측 성공률</th><th>수치 기준 확인</th></tr>'
    for check in prediction_summaries:
        accuracy=f"{check['accuracy_pct']:.1f}%" if check['accuracy_pct'] is not None else 'Not measured'
        prediction_table+='<tr>'+''.join('<td>'+esc(x)+'</td>' for x in
            (check['mode'],f"{check['successful_predictions']}/{check['observations']}",accuracy,labels[check['numerical_status']]))+'</tr>'
    prediction_section=f'''<section><h2>예측오차 수치 확인 · TTA 통과 미확인</h2>
<p>평가 라운드의 원본 예측값과 실제 학습시간으로 상대오차를 계산했습니다. 상대오차 10% 이하가 성공이며 성공률 목표는 90%입니다.</p>
<div class="scroll"><table>{prediction_table}</table></div>
<p class="note">예측 시점·데이터 독립성 증빙은 이 비교에서 검증하지 않았습니다. 이 조건의 사전 SLO가 없어 TC4는 미측정입니다. 기존 TTA 시험의 PASS를 이 비교에 적용할 수 없습니다.</p>
<p><a href="prediction_check.csv">기기·라운드별 예측값, 실제값 및 상대오차 CSV</a></p></section>'''
    page=page.replace('<section><h2>평가 라운드 합계',prediction_section+'<section><h2>평가 라운드 합계',1)
    (out/'index.html').write_text(page)
    return out


def workload_comparison(runs,out=None):
    items=[]
    for run in runs:
        page=single_report(run)
        summary=read(page.parent/'summary.json');condition=summary['condition']
        work=rows(page.parent/'policy_rounds.csv')
        state=read(Path(run)/'state.json')
        check=summary['prediction_check']
        items.append({'run':str(Path(run).resolve()),'page':str(page),
            'condition':condition,'pi_quota':condition['config']['rpi-normal-quota'],
            'nx_base':condition['config']['jetson-base-quota'],
            'nx_max':condition['config']['jetson-max-quota'],
            'nx_step':condition['config']['jetson-sample-step'],
            'nx_samples_by_round':{r['round']:int(r['selected_samples']) for r in work if r['device_name']=='jetson_orin_nx'},
            'prediction_check':check,'mare_pct':100-summary['one_minus_mare_pct'] if summary['one_minus_mare_pct'] is not None else None,
            'attempt_s':state['end']-state['start']})
    for key in ('model','dataset','manifest_sha256','seed','rounds','mode'):
        if len({str(item['condition'][key]) for item in items})!=1:raise ValueError('workload comparison differs: '+key)
    for key in ('max-seq-len','batch_size','local-epochs','lora-r','sample-selection-mode','residual-calibration-enabled'):
        if len({str(item['condition']['config'][key]) for item in items})!=1:raise ValueError('workload comparison differs: '+key)
    out=Path(out or OUT/('workloads_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps({'runs':items,'tta_pass':None},ensure_ascii=False,indent=2))
    esc=lambda x:html.escape(str(x),quote=True)
    body=''
    for item in items:
        check=item['prediction_check'];mare=item['mare_pct']
        values=(item['pi_quota'],f"{item['nx_base']}–{item['nx_max']} / 간격 {item['nx_step']}",
                ' → '.join(str(v) for k,v in sorted(item['nx_samples_by_round'].items(),key=lambda kv:int(kv[0]))),
                f"{check['successful_predictions']}/{check['observations']}",f'{mare:.2f}%' if mare is not None else 'Not measured',
                check['prediction_source'],f"{item['attempt_s']/60:.2f}분")
        link=esc(os.path.relpath(item['page'],out))
        body+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+f'<td><a href="{link}">상세 결과</a></td></tr>'
    header=''.join('<th>'+esc(v)+'</th>' for v in ('Pi 샘플','NX 후보 범위','NX 실제 샘플 R1→R5','오차≤10%','평균 상대오차','예측 출처','준비 포함 시간','결과'))
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LoRACast 작업량 확대 시험</title>
<style>body{{font:15px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:20px;color:#193047}}table{{border-collapse:collapse;width:100%}}th,td{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow:auto}}</style>
<h1>LoRACast 작업량 확대 시험</h1><p>{esc(items[0]['condition']['model'])} · {esc(items[0]['condition']['dataset'])} · L={items[0]['condition']['config']['max-seq-len']} · 처음 2회 보정, 이후 3회 평가</p>
<div class="scroll"><table><tr>{header}</tr>{body}</table></div>
<p>before_dispatch는 학습 요청 전에 저장한 예측이며 post_reply_history_only는 기존 CSV의 응답 후 재계산값입니다. 모든 평가 라운드를 포함했습니다. 작업량과 Jetson 후보 범위를 함께 변경했으므로 각 요인의 독립적인 효과를 증명하는 실험은 아닙니다.</p>
<p>TTA 전체 통과·에너지 절감 판정은 하지 않았습니다. <a href="summary.json">설정·원본 경로·요약 JSON</a></p></html>'''
    (out/'index.html').write_text(page)
    return out


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--workloads',action='store_true',help='같은 정책에서 작업량 크기를 비교')
    p.add_argument('--out',type=Path)
    a=p.parse_args(argv)
    try:
        output=workload_comparison(a.runs,a.out) if a.workloads else compare(a.runs,a.out)
        print('Comparison:',output/'index.html')
        if not a.workloads:print('CSV:',output/'policy_comparison.csv')
        return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print('ERROR:',exc,file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
