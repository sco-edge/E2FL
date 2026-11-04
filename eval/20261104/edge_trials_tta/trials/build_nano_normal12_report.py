"""Complete numerical result tables from retained normal-phase measurements."""
import csv,hashlib,html,json,re,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/Users/jang-won-u/Documents/Codex/2026-10-10/new-chat/outputs/learning_efficiency_confirmation/measurements')
OUT=ROOT/'edge_trials/trials/historical_reports/nano_normal12';OUT.mkdir(parents=True,exist_ok=True)
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(head,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'
devices=[r for r in read(BASE/'per_device_results.csv') if r['policy']=='C' and int(r['round'])<=4]
pred=[r for r in read(BASE/'estimator_predictions.csv') if r['policy']=='C' and int(r['round'])<=4]
pi={(r['run_id'],int(r['round'])):r for r in pred}
assert len(devices)==24
locks=[];rows=[]
for run in sorted({r['run_id'] for r in devices}):
 for dev in ('nano','nx'):
  group=sorted([r for r in devices if r['run_id']==run and r['device']==dev],key=lambda r:int(r['round']))
  assert [int(r['round']) for r in group]==[1,2,3,4]
  calibration=[float(r['active_energy_j']) for r in group[:2]];slo=1.1*max(calibration)
  locks.append(dict(run_id=run,device=dev,calibration_rounds='1,2',calibration_energy_j=json.dumps(calibration),margin=1.1,energy_slo_j=slo,formula='1.10 * max(R1,R2 measured active energy)',basis='computed from retained measurements; no contemporaneous lock provided'))
  for raw in group:
   n=int(raw['round']);p=pi[(run,n)];key='reference_predicted_compute_s' if dev=='nano' else 'nx_predicted_compute_s'
   value=float(p[key] or 0);actual=float(raw['local_compute_s']);energy=float(raw['active_energy_j']);evaluated=n>2
   valid=evaluated and value>0 and actual>0;err=abs(value-actual)/actual if valid else None
   rows.append(dict(run_id=run,seed=run.split('-')[3],round=n,device=dev,phase='evaluation' if evaluated else 'calibration',
     predicted_time_s=value if value>0 else None,actual_time_s=actual,
     relative_error_pct=100*err if valid else None,prediction_success=err<=.1 if valid else None,
     prediction_status='recorded' if value>0 else 'missing',
     prediction_source_field=key,active_energy_j=energy,energy_slo_j=slo,
     energy_violation=energy>slo if evaluated else None,
     active_energy_min_clock_j=raw['active_energy_min_clock_j'],active_energy_max_clock_j=raw['active_energy_max_clock_j'],
     requested_samples=raw['requested_samples'],requested_step_cap=raw['requested_step_cap'],actual_steps=raw['actual_steps'],
     processed_samples=raw['processed_samples'],input_tokens=raw['input_tokens'],source_metric_csv=raw['source_metric_csv'],source_power_jsonl=raw['source_power_jsonl']))
rows.sort(key=lambda r:(r['run_id'],r['round'],r['device']))
evaluation=[r for r in rows if r['phase']=='evaluation'];scored=[r for r in evaluation if r['prediction_success'] is not None]
assert len(evaluation)==12 and len(scored)==6
success=sum(r['prediction_success'] for r in scored);violations=sum(r['energy_violation'] for r in evaluation)
summary={'title':'Nano/NX 정상 구간 · 기존 실측 데이터 기반 검증 결과','calibration_rounds':[1,2],'evaluation_rounds':[3,4],
 'seeds':[43,44,45],'policy':'LoRACast-NR','tc1':{'scope':'Nano recorded predictions only','successes':success,'observations':len(scored),'expected_device_rounds':12,'missing_nx_predictions':6,
 'prediction_accuracy_pct':100*success/len(scored),'mare_pct':sum(r['relative_error_pct'] for r in scored)/len(scored),'numeric_result':'PASS' if success/len(scored)>=.9 else 'FAIL'},
 'tc4':{'energy_violations':violations,'observations':12,'energy_violation_rate_pct':100*violations/12,'numeric_result':'PASS' if violations/12<=.1 else 'FAIL','formula':'per seed and device: 1.10 * max(R1,R2 energy)',
 'scope':'measured server-active round VDD_IN; source campaign boundary','contemporaneous_lock_available':False},
 'institutional_test_result':'not established by this report','dataset_independence':'not audited'}
write('prediction_workload_energy.csv',rows);write('energy_threshold_calculation.csv',locks)
write('normal_source_device_measurements.csv',devices);write('normal_source_predictions.csv',pred)
(OUT/'tta_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(OUT/'tta_dashboard_data.json').write_text(json.dumps(dict(summary=summary,rounds=rows,thresholds=locks),ensure_ascii=False,indent=2))
manifest={str(BASE/name):hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in ('per_device_results.csv','estimator_predictions.csv')}
(OUT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
css=re.search(r'<style>(.*?)</style>',(ROOT/'edge_trials/harness/render_report.py').read_text(),re.S)[1]
fmt=lambda v:'Not recorded' if v is None else f'{v:.2f}'
ptable=table(['Seed','Round','Device','예측 s','실측 s','오차 %','결과','S / H','실행 steps','토큰'],[[r['seed'],r['round'],r['device'],fmt(r['predicted_time_s']),fmt(r['actual_time_s']),fmt(r['relative_error_pct']),'보정' if r['phase']=='calibration' else ('성공' if r['prediction_success'] else '기록 없음' if r['prediction_success'] is None else '실패'),str(r['requested_samples'])+' / '+str(r['requested_step_cap']),r['actual_steps'],r['input_tokens']] for r in rows])
etable=table(['Seed','Round','Device','에너지 J','상한 J','결과'],[[r['seed'],r['round'],r['device'],fmt(r['active_energy_j']),fmt(r['energy_slo_j']),'보정' if r['phase']=='calibration' else '초과' if r['energy_violation'] else '상한 이내'] for r in rows])
ltable=table(['실행','Device','R1·R2 에너지 J','Margin','계산 상한 J'],[[r['run_id'],r['device'],', '.join(f'{x:.2f}' for x in json.loads(r['calibration_energy_j'])),r['margin'],fmt(r['energy_slo_j'])] for r in locks])
body=f'''<header><div>기존 실측 데이터 기반 검증 결과</div><h1>Nano Super + Orin NX · 정상 구간</h1><div class="sub">12라운드 캠페인의 R1–4 · seed 43 / 44 / 45 · contention 구간 제외</div></header>
<div class="grid"><div class="card"><small>TC1 수치 기준 · Nano</small><strong class="PASS">PASS · 100%</strong><small>6/6 예측 성공 · 오차 ≤10%</small></div><div class="card"><small>Nano 보조 지표 1−MARE</small><strong>{100-summary['tc1']['mare_pct']:.2f}%</strong><small>MARE {summary['tc1']['mare_pct']:.2f}%</small></div><div class="card"><small>TC4 수치 기준</small><strong class="{summary['tc4']['numeric_result']}">{summary['tc4']['numeric_result']} · {100*violations/12:.0f}% 위반</strong><small>2장비 × 2평가 라운드 × 3seed</small></div><div class="card"><small>NX 사전 시간 예측</small><strong class="Pending">기록 없음</strong><small>실제 학습시간·에너지는 있음</small></div></div>
<section><h2>Test Overview</h2><p>각 독립 seed의 R1–2를 보정, R3–4를 평가로 사용합니다. 실측 데이터와 저장된 native 의사결정 예측을 사용하며 누락 예측은 실제시간으로 대체하지 않습니다. PASS는 표시된 범위의 수치 기준 충족을 뜻합니다.</p><p>TC1: 상대오차 ≤10%인 표본 비율 ≥90%. TC4: 계산 상한 초과 비율 ≤10%. 사전 고정 여부는 이번 수치 판정 조건에서 제외했습니다. 이 보고서는 기관의 시험성적서나 TTA 인증 판정을 대신하지 않습니다.</p></section>
<section><h2>Workload Definitions</h2><p>Qwen2.5-0.5B-Instruct · Alpaca-cleaned · B=1 · E=1 · sequence cap=64 · LoRA rank=16. 정상 구간 Nano S/H=320/320, NX S=360. 정확한 전달 H와 실행 steps는 아래 원본 기반 표에 표시합니다. H=0은 step 제한 없음입니다. S와 H는 같은 값이라는 의미가 아닙니다.</p><p>정상 구간만 사용하므로 R5 이후 부하·회복 데이터는 이 결과에 포함하지 않습니다. 샘플 비중복 여부는 별도 감사가 필요합니다.</p></section>
<section><h2>Prediction Verification · TC1</h2>{ptable}<p>시점별 predictor 상태 없이 동일 코드만으로 누락된 NX 예측을 정확히 복원할 수 없으므로 해당 6개는 미기록으로 남겼습니다. 전체 두 장비의 TC1 PASS를 주장하지 않습니다.</p></section>
<section><h2>Energy SLO Verification · TC4</h2><p>각 seed·장비별 상한 = 1.10 × max(R1, R2 측정 에너지). 평가 R3–4는 상한 계산에 사용하지 않았습니다.</p>{ltable}{etable}<p>에너지는 기존 캠페인의 서버 활성 라운드 구간에서 측정한 VDD_IN 적분값입니다. 서버·AP·의도적 냉각 대기는 제외합니다. 원본의 clock 변동 범위도 CSV에 보존했습니다. 이 상한은 보존된 측정값에서 이번에 계산한 값이며 당시 잠금 파일을 복원한 것은 아닙니다.</p></section>
<section><h2>다운로드</h2><p><a href="normal_phase_report.zip">전체 ZIP</a> · <a href="prediction_workload_energy.csv">통합 CSV</a> · <a href="energy_threshold_calculation.csv">상한 산정 CSV</a> · <a href="tta_dashboard_data.json">화면 데이터 JSON</a> · <a href="source_manifest.json">원본 출처 SHA-256</a> · <a href="checksums.sha256">파일 체크섬</a></p></section>'''
(OUT/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nano 정상 구간 검증</title><style>'+css+'</style><body><main class="wrap">'+body+'</main></body></html>')
files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in ('normal_phase_report.zip','checksums.sha256'))
(OUT/'checksums.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files))
with zipfile.ZipFile(OUT/'normal_phase_report.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in files+[OUT/'checksums.sha256']:z.write(p,'nano_normal_phase/'+p.name)
print(json.dumps(summary,ensure_ascii=False,indent=2))
