"""Reproduce retrospective TC1 reports from hash-verified October hardware CSVs."""
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'workload_policy'
OUT=ROOT/'edge_trials/trials/historical_reports/october_20rounds'

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((SOURCE/'source_manifest.json').read_text())
    summaries=[];all_rows=[];sources=[]
    for arm,label in [('nr','Rebalancing OFF'),('full','Rebalancing ON')]:
        path=next((SOURCE/'inputs/baselines'/arm).glob('fedavg*.csv'))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==manifest['files_sha256'][str(path.relative_to(SOURCE))]
        raw=list(csv.DictReader(path.open()))
        enriched=[];train=[];seen=set()
        for row in raw:
            record={'experiment':label,'round':row['round'],'device_name':row['device_name'],
                    'phase':row['phase'],'evaluation_status':'not_training',
                    'predicted_time_s':'','actual_time_s':'','absolute_error_s':'',
                    'relative_error_pct':'','prediction_success':'','one_minus_relative_error_pct':'',
                    'energy_measurement_status':'not_measured','energy_slo_status':'not_found',
                    'prediction_timing_audit':'not_verified',**row}
            if row['phase']=='train':
                key=(row['round'],row['device_name']);assert key not in seen;seen.add(key)
                pred=float(row['pred_train_time_self']);actual=float(row['actual_train_time_self'])
                record.update(predicted_time_s=pred,actual_time_s=actual,
                              evaluation_status='warmup_excluded' if int(row['round'])<=2 else 'evaluated')
                if int(row['round'])>2:
                    assert all(math.isfinite(v) and v>0 for v in (pred,actual))
                    error=abs(pred-actual)/actual
                    record.update(absolute_error_s=abs(pred-actual),relative_error_pct=100*error,
                                  prediction_success=error<=.1,one_minus_relative_error_pct=100*(1-error))
                train.append(record)
            enriched.append(record)
        train.sort(key=lambda r:(int(r['round']),r['device_name']))
        for device in ('overall','RPi5','jetson_orin_nx'):
            rows=[r for r in train if r['evaluation_status']=='evaluated' and (device=='overall' or r['device_name']==device)]
            n=len(rows);success=sum(r['prediction_success'] for r in rows)
            mare=sum(r['relative_error_pct'] for r in rows)/n
            summaries.append(dict(experiment=label,device=device,evaluated_device_rounds=n,
                successful_predictions=success,prediction_accuracy_pct=100*success/n,
                mare_pct=mare,one_minus_mare_pct=100-mare,
                time_error_limit_pct=10,target_accuracy_pct=90,
                tc1_numeric_result='PASS' if success/n>=.9 else 'FAIL',
                verdict_scope='retrospective CSV numeric criterion only',
                full_tta_result='not_verified',energy_status='not_measured'))
        write_csv(OUT/f'{arm}_updated.csv',enriched)
        write_csv(OUT/f'{arm}_prediction_results.csv',train)
        shutil.copy2(path,OUT/f'{arm}_original.csv')
        config=next((SOURCE/'inputs').glob(f'cooled-strong-{"nr" if arm=="nr" else "rb"}-*.toml'))
        shutil.copy2(config,OUT/f'{arm}_config.toml')
        sources.append(dict(experiment=label,path=str(path),sha256=digest,hash_verified=True,
                            energy_values=sorted({r['energy'] for r in raw if r['phase']=='train'}),
                            power_monitor=False))
        all_rows.extend(train)
    write_csv(OUT/'accuracy_summary.csv',summaries)
    energy={'status':'not_measured','whole_round_energy_available':False,
            'precommitted_slo_found':False,'sources':sources,
            'basis':'Both source configs set power-monitor=false. All train energy fields are 0.0; these are not measured zero energy.',
            'separate_measurement_reference':str(ROOT/'exposure_policy/client_window_energy_assessment.md'),
            'limitations':'Separate October power profiles and legacy power-times-duration estimates do not establish energy or SLO compliance for these two runs.'}
    (OUT/'energy_evidence.json').write_text(json.dumps(energy,ensure_ascii=False,indent=2))
    (OUT/'summary.json').write_text(json.dumps(dict(summaries=summaries,sources=sources,
        excluded_rounds=[1,2],energy=energy,prediction_timing_audit='not_verified'),ensure_ascii=False,indent=2))
    cards=''
    for arm,label in [('nr','Rebalancing OFF'),('full','Rebalancing ON')]:
        s=next(x for x in summaries if x['experiment']==label and x['device']=='overall')
        cards+=f'''<section><h2>{label}</h2><span class="pass">TC1 수치 기준 {s['tc1_numeric_result']}</span>
        <div class="big">{s['prediction_accuracy_pct']:.2f}%</div>
        <p>성공 {s['successful_predictions']} / {s['evaluated_device_rounds']}개 · 목표 90% 이상</p>
        <p>MARE {s['mare_pct']:.2f}% · 1−MARE {s['one_minus_mare_pct']:.2f}%</p>
        <p><a href="{arm}_updated.csv" download>원본 열 포함 확장 CSV</a> · <a href="{arm}_prediction_results.csv" download>학습 예측 결과 CSV</a></p></section>'''
    statrows=''.join(f"<tr><td>{s['experiment']}</td><td>{s['device']}</td><td>{s['successful_predictions']}/{s['evaluated_device_rounds']}</td><td>{s['prediction_accuracy_pct']:.2f}%</td><td>{s['mare_pct']:.2f}%</td><td>{s['one_minus_mare_pct']:.2f}%</td></tr>" for s in summaries)
    rows=''
    for r in all_rows:
        evaluated=r['evaluation_status']=='evaluated'
        result=('성공' if r['prediction_success'] else '실패') if evaluated else '보정 · 평가 제외'
        error=f"{r['relative_error_pct']:.2f}%" if evaluated else '—'
        css='pass' if evaluated and r['prediction_success'] else 'fail' if evaluated else 'muted'
        rows+=f'''<tr data-arm="{r['experiment']}" data-result="{result}"><td>{r['experiment']}</td><td>{r['round']}</td><td>{r['device_name']}</td><td>{r['predicted_time_s']:.3f}</td><td>{r['actual_time_s']:.3f}</td><td>{error}</td><td class="{css}">{result}</td><td>{r['num_examples']}</td><td>{r['input_tokens']}</td><td>{r['seq_len']}</td></tr>'''
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>10월 실제 장비 20라운드 · LoRACast</title>
    <style>body{font:16px system-ui,sans-serif;background:#f3f6fa;color:#182637;margin:0}main{max-width:1180px;margin:auto;padding:30px 22px}h1{font-size:30px}h2{font-size:21px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}section{background:white;padding:24px;border:1px solid #dde4ed;border-radius:14px;margin:16px 0}.big{font-size:42px;font-weight:750;margin-top:14px}.pass{color:#147347;font-weight:650}.fail{color:#b52e35;font-weight:650}.muted{color:#697789}a{color:#175cc3}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:11px;text-align:left;border-bottom:1px solid #e4e9ef;white-space:nowrap}th{background:#edf2f8}.scroll{overflow:auto}select{padding:9px;margin:10px 8px 15px 0}p{line-height:1.65}.note{background:#fff6df;padding:16px;border-radius:10px}</style><main>
    <p class="muted">2026년 10월 12일 · 실제 Raspberry Pi 5 + Jetson Orin NX</p><h1>LoRACast 20라운드 예측 검증</h1>
    <p>초기 R1–R2 제외 · R3–R20의 실제 학습 참여 표본 평가 · 원본 CSV SHA-256 확인 완료</p>
    <p class="note">PASS는 기존 CSV에 기록된 예측값의 TC1 수치 기준 판정입니다. 학습 전 예측 생성 시점은 이번 보고서에서 검증하지 않았으며, 에너지·SLO 및 TTA 종합 판정은 미검증입니다.</p>
    <div class="cards">CARDS</div>
    <section><h2>판정 기준과 실험 조건</h2><p>상대오차 = |예측시간 − 실제시간| / 실제시간. 오차 ≤ 10%이면 성공이며, 전체 유효 device-round의 성공률 ≥ 90%이면 TC1 수치 기준 PASS입니다. 1−MARE는 보조 지표입니다. 장비별 결과는 참고용이며 전체 표본으로 주 지표를 계산합니다.</p>
    <p>Pi UID: rpi55 · NX UID: jetson-orin-nx-1 · Pi 샘플 4 · NX 8→32 · 최대 길이 64 · B=1 · E=1 · round_shuffle, seed 42. Rebalancing ON은 NX 학습 생략 라운드를 분모에서 제외합니다. 과거 participation rebalancing이며 현재 Exposure TBM과 동일한 옵션으로 해석하지 않습니다.</p>
    <div class="scroll"><table><tr><th>조건</th><th>장비</th><th>성공/평가</th><th>성공률</th><th>MARE</th><th>1−MARE</th></tr>STATS</table></div></section>
    <section><h2>에너지 증빙</h2><p><strong>Not measured — 두 실행의 전체 라운드 에너지와 사전 고정 SLO를 확인할 수 없습니다.</strong></p><p>두 설정은 power-monitor=false이며 학습 CSV의 energy 값은 모두 0.0입니다. 이를 소비 에너지 0 J로 표시하거나 PASS로 판정하지 않습니다. 별도 시점의 전력 프로파일을 곱한 추정값도 이 실행의 실측값으로 대체하지 않았습니다.</p><a href="energy_evidence.json">확인 근거 JSON</a></section>
    <section><h2>라운드별 예측과 측정</h2><label>조건 <select id="arm"><option value="">전체</option><option>Rebalancing OFF</option><option>Rebalancing ON</option></select></label><label>결과 <select id="result"><option value="">전체</option><option>성공</option><option>실패</option><option>보정 · 평가 제외</option></select></label>
    <div class="scroll"><table><thead><tr><th>조건</th><th>Round</th><th>장비</th><th>예측 (s)</th><th>실제 (s)</th><th>상대오차</th><th>결과</th><th>샘플</th><th>토큰</th><th>길이 상한</th></tr></thead><tbody id="details">ROWS</tbody></table></div></section>
    <section><h2>다운로드</h2><p><a href="october_20rounds_report.zip">전체 결과 ZIP</a> · <a href="accuracy_summary.csv">정확도 요약 CSV</a> · <a href="summary.json">요약 JSON</a> · <a href="checksums.sha256">파일 SHA-256</a></p><p>확장 CSV는 원본 모든 행·열을 유지하며 분석 열을 추가했습니다. 학습 결과 CSV는 학습 행만 모았습니다. 원본 사본과 당시 설정도 ZIP에 포함되어 있습니다. CSV의 원래 accuracy 열은 예측 성공률이 아니므로 prediction_success와 relative_error_pct를 확인하세요.</p></section>
    </main><script>const a=document.getElementById('arm'),b=document.getElementById('result');function filter(){document.querySelectorAll('#details tr').forEach(r=>r.hidden=!!((a.value&&r.dataset.arm!==a.value)||(b.value&&r.dataset.result!==b.value)))}a.onchange=b.onchange=filter;</script></html>'''
    (OUT/'index.html').write_text(page.replace('CARDS',cards).replace('STATS',statrows).replace('ROWS',rows))
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in ('checksums.sha256','october_20rounds_report.zip'))
    (OUT/'checksums.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files))
    with zipfile.ZipFile(OUT/'october_20rounds_report.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[OUT/'checksums.sha256']:z.write(p,'october_20rounds/'+p.name)
    assert [(s['evaluated_device_rounds'],s['successful_predictions']) for s in summaries if s['device']=='overall']==[(36,33),(27,27)]
    print(OUT/'index.html')

if __name__=='__main__':build()
