"""Build a standalone onsite-style retrospective report, without inventing SLO evidence."""
import csv, hashlib, html, json, math, re, zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent if (Path(__file__).resolve().parent/'raw').is_dir() else ROOT/'edge_trials/trials/historical_reports/nano_normal6'
RAW=OUT/'raw'

def records(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def dump_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def integrate(points,start,end):
    if not points or points[0][0]>start or points[-1][0]<end:raise ValueError('Power trace does not cover boundaries')
    total=0.;gaps=[]
    for (a,p),(b,q) in zip(points,points[1:]):
        if b<=a:raise ValueError('Power timestamps not increasing')
        if b<=start or a>=end:continue
        gaps.append(b-a)
        if b-a>3:raise ValueError('Power gap exceeds 3 s')
        left=max(a,start);right=min(b,end)
        pl=p+(q-p)*(left-a)/(b-a);pr=p+(q-p)*(right-a)/(b-a)
        total+=(pl+pr)/2*(right-left)
    return total,max(gaps),sum(start<=t<=end for t,p in points)
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def stamp(t):return datetime.fromtimestamp(t,ZoneInfo('Asia/Seoul')).strftime('%H:%M:%S')
def build():
    source=json.loads((RAW/'predictions_and_training.json').read_text())
    decisions={d['round']:d for d in source['decisions']}
    train=sorted([r for r in source['csv'] if r['phase']=='train'],key=lambda r:(int(r['round']),r['device_name']))
    assert len(train)==12 and len({(r['round'],r['device_name']) for r in train})==12
    events=records(RAW/'reply_arrivals.jsonl');ends={e['call']:e for e in events if e['event']=='all_replies_returned'}
    pushes=[e for e in events if e['event']=='push_end' and all(x['message_type']=='train' for x in e['requests'])]
    assert len(pushes)==6
    boundaries={}
    for i,e in enumerate(pushes,1):
        assert ends[e['call']]['reply_count']==2
        boundaries[i]=(e['server_unix_s'],ends[e['call']]['server_unix_s'])
    power={};power_rows=[]
    for dev,name in [('jetson_orin_nano_super','nano'),('jetson_orin_nx','nx')]:
        points=[]
        for r in records(RAW/f'{name}_telemetry.jsonl'):
            match=re.search(r'\bVDD_IN\s+(\d+(?:\.\d+)?)mW/',r['raw'])
            if not match:continue
            p=float(match[1])/1000;t=float(r['unix_s']);points.append((t,p))
            power_rows.append(dict(device_name=dev,unix_time_s=t,power_w=p,source=f'raw/{name}_telemetry.jsonl'))
        power[dev]=points
    output=[];energies=[]
    for r in train:
        n=int(r['round']);dev=r['device_name'];d=decisions[n];request=d['requested'][dev]
        nano=dev=='jetson_orin_nano_super'
        field='reference_predicted_seconds' if nano else ('base_nx_predicted_seconds' if d.get('normal_fallback_reason')=='base_candidate_outside_compute_window' else 'normal_nx_predicted_seconds')
        p=d[field];a=float(r['actual_train_time_self']);evaluated=n>2
        error=abs(p-a)/a if evaluated else None
        assert not evaluated or (math.isfinite(p) and p>0 and a>0)
        start,end=boundaries[n]
        try:joules,gap,count=integrate(power[dev],start,end);coverage='covered; clock alignment unverified'
        except ValueError as exc:joules=gap=count=None;coverage=str(exc)
        energy=dict(round=n,device_name=dev,round_start_timestamp=start,round_end_timestamp=end,
                    round_duration_s=end-start,power_samples=count,max_power_sample_gap_s=gap,
                    barrier_energy_j=joules,energy_slo_j=None,energy_violation=None,
                    clock_alignment_verified=False,coverage_status=coverage,
                    measurement_scope='train push_end to all training replies; excludes evaluation, aggregation, cooldown')
        energies.append(energy)
        output.append(dict(round=n,device_name=dev,device_uid=r['device_uid'],
            phase='evaluation' if evaluated else 'warmup',predicted_time_s=p,actual_time_s=a,
            absolute_error_s=abs(p-a) if evaluated else None,relative_error_pct=100*error if evaluated else None,
            prediction_success=error<=.1 if evaluated else None,
            one_minus_relative_error_pct=100*(1-error) if evaluated else None,
            prediction_source_field=field,csv_logged_prediction_s=r['pred_train_time_self'],
            requested_samples=request['samples'],requested_step_cap=request['steps'],
            executed_samples=int(r['num_examples']),executed_steps=r['local_steps'],
            local_epochs=r['local_epochs'],batch_size=r['batch_size'],max_seq_len=r['seq_len'],
            input_tokens=r['input_tokens'],lora_rank=r['lora_rank'],lora_modules=r['lora_modules'] or 'Not recorded',
            beta_logged=r['est_beta_self'] or 'Not recorded',alpha_logged=r['est_alpha_self'],
            nx_fallback_reason=d.get('normal_fallback_reason') if not nano else None,
            **{k:v for k,v in energy.items() if k not in ('round','device_name')}))
    summaries=[]
    for dev in ('overall','jetson_orin_nano_super','jetson_orin_nx'):
        rows=[r for r in output if r['phase']=='evaluation' and (dev=='overall' or r['device_name']==dev)]
        successes=sum(r['prediction_success'] for r in rows);mare=sum(r['relative_error_pct'] for r in rows)/len(rows)
        summaries.append(dict(device_name=dev,observations=len(rows),successes=successes,
            prediction_accuracy_pct=100*successes/len(rows),mare_pct=mare,one_minus_mare_pct=100-mare,
            tc1_numeric_result='PASS' if successes/len(rows)>=.9 else 'FAIL'))
    summary=dict(experiment='nano-nx-unlocked-n320-x336to480-s42-r6-20261021',total_rounds=6,warmup_rounds=2,
        measured_rounds=4,devices=summaries,tc1_numeric_result=summaries[0]['tc1_numeric_result'],
        tc4_result='Not verified: no precommitted SLO found',full_tta_result='Not verified',
        energy_status='retrospective integral of measured VDD_IN; clock alignment unverified',
        energy_scope=energies[0]['measurement_scope'],dataset_independence='Not audited',
        predictor_audit='Native decision records; no onsite timestamp/coefficient audit',
        energy_by_device={dev:sum(e['barrier_energy_j'] for e in energies if e['device_name']==dev and e['barrier_energy_j'] is not None) for dev in power})
    dump_csv('sample_workload_prediction_energy.csv',output);dump_csv('tta_energy_audit.csv',energies)
    dump_csv('accuracy_summary.csv',summaries);dump_csv('power_samples.csv',power_rows);dump_csv('original_fedavg.csv',source['csv'])
    data=dict(summary=summary,rounds=output,energy=energies)
    (OUT/'tta_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (OUT/'tta_dashboard_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    css=(OUT/'onsite_style.css').read_text() if (OUT/'onsite_style.css').is_file() else re.search(r'<style>(.*?)</style>',(ROOT/'edge_trials/harness/render_report.py').read_text(),re.S)[1]
    (OUT/'onsite_style.css').write_text(css)
    if Path(__file__).resolve()!=OUT/'build_nano_normal_report.py':
        (OUT/'build_nano_normal_report.py').write_text(Path(__file__).read_text())
    label=lambda d:'Nano Super' if 'nano' in d else 'Orin NX'
    ptable=table(['Round','Device / UID','구간','예측 s','실제 s','상대오차','결과','S / H','실행 샘플 / steps','토큰'],[
        [r['round'],label(r['device_name'])+' / '+r['device_uid'],'평가' if r['phase']=='evaluation' else '보정',f"{r['predicted_time_s']:.3f}",f"{r['actual_time_s']:.3f}",f"{r['relative_error_pct']:.2f}%" if r['relative_error_pct'] is not None else '—','성공' if r['prediction_success'] else '보정 제외' if r['prediction_success'] is None else '실패',f"{r['requested_samples']} / {r['requested_step_cap']}",f"{r['executed_samples']} / {r['executed_steps']}",r['input_tokens']] for r in output])
    etable=table(['Round','Device','시작–종료 KST','구간 s','전력 샘플','최대 간격 s','에너지 J (참고)','사전 SLO','위반 판정'],[
        [e['round'],label(e['device_name']),stamp(e['round_start_timestamp'])+'–'+stamp(e['round_end_timestamp']),f"{e['round_duration_s']:.2f}",e['power_samples'],f"{e['max_power_sample_gap_s']:.3f}" if e['max_power_sample_gap_s'] else '미확인',f"{e['barrier_energy_j']:.1f}" if e['barrier_energy_j'] is not None else '계산 보류','Not found','Not verified'] for e in energies])
    defs=table(['항목','정의 / 역할','Nano Super','Orin NX'],[
        ['Sample cap S','최대 허용 샘플 수; predictor의 작업량 입력',320,'336–352'],
        ['Local steps H','Optimizer step 상한; 0은 제한 없음',320,'0 또는 335–349'],
        ['Epoch E','학습 반복 횟수; 작업량 입력',1,1],['Batch B','미니배치 크기',1,1],
        ['Sequence cap L','최대 입력 길이; 실제 평균 토큰 길이와 다름',64,64],
        ['LoRA rank','저랭크 차원; 기록된 환경 설정',16,16],
        ['LoRA modules','원본 CSV에 모듈 목록 없음','Not recorded','Not recorded'],
        ['Model / Dataset','실험 보고서·고정 설정 기준','Qwen2.5-0.5B-Instruct / Alpaca-cleaned','동일'],
        ['Sample selection','Strong partition, round_shuffle, seed 42','샘플 비중복 감사 없음','샘플 비중복 감사 없음']])
    bars=''
    for r in output:
        if r['phase']!='evaluation':continue
        bars+=f'<div class="barrow"><span>R{r["round"]} {label(r["device_name"])}</span><div><div class="track"><div class="fill" style="width:{r["predicted_time_s"]/120*100}%"></div></div><div class="track"><div class="fill actual" style="width:{r["actual_time_s"]/120*100}%"></div></div></div><span>{r["predicted_time_s"]:.1f} / {r["actual_time_s"]:.1f}</span></div>'
    s=summaries[0]
    body=f'''<header><div>과거 실장비 결과 · TC1 / 에너지 증빙</div><h1>Nano Super + Orin NX · 정상 6라운드</h1><div class="sub">2026-10-21 · 배경 부하 없음 · 보정 2 + 평가 4라운드 · 6/6 완료</div></header>
    <div class="grid"><div class="card"><small>TC1 수치 기준</small><strong class="PASS">{s['tc1_numeric_result']} · {s['prediction_accuracy_pct']:.0f}%</strong><small>8/8 성공 · 목표 90%</small></div><div class="card"><small>보조 지표 1−MARE</small><strong>{s['one_minus_mare_pct']:.2f}%</strong><small>MARE {s['mare_pct']:.2f}%</small></div><div class="card"><small>에너지 측정 원본</small><strong>VDD_IN 있음</strong><small>시계 정렬 미검증 · 사후 구간 적분</small></div><div class="card"><small>TC4 / TTA 종합</small><strong class="Pending">Not verified</strong><small>사전 고정 SLO 미확인</small></div></div>
    <section><h2>Test Overview</h2><p>onsite와 같은 화면 구성을 사용하는 과거 실험 보고서입니다. 현재 onsite 시험을 새로 수행한 결과는 아닙니다. 당시 정상 6라운드 결과만 포함하며 contention 실험은 포함하지 않습니다.</p><p>Native NanoReferenceNR 가변 할당 · rebalancing OFF · seed 42 · Qwen2.5-0.5B-Instruct · Alpaca-cleaned. 원본 실행 QC는 12개 학습 응답, 7개 체크포인트, 온도·메모리 가드 통과를 기록합니다.</p></section>
    <section><h2>Workload Definitions</h2>{defs}<p class="hint">위 표의 범위는 전체 6라운드 기준입니다. 정확한 라운드별 전달 S/H는 아래 표와 통합 CSV에서 확인하세요.</p></section>
    <section><h2>Prediction Verification · TC1</h2><p>|예측−실제| / 실제 ≤ 10%인 device-round의 비율로 판정합니다. R1–2는 제외하고 R3–6의 8개 표본을 평가했습니다.</p><div class="bars">{bars}</div><p class="hint">파랑: 사전 의사결정 예측 · 주황: 실제 로컬 학습시간 (초)</p>{ptable}<p class="hint">NX R3–4는 기본 후보가 목표 시간 창을 넘겨 fallback한 라운드입니다. 실제 실행한 기본 작업량에 대응하는 base_nx_predicted_seconds를 사용했습니다. 그 외 NX는 normal_nx_predicted_seconds, Nano는 reference_predicted_seconds입니다. CSV의 사후 기록 예측값도 통합 CSV에 별도 보존했습니다.</p><p>예측 시각·계수 갱신 시각의 onsite 감사와 보정/평가 샘플 ID 비중복은 미검증입니다. 개별 샘플 ID가 없는 상태에서 서로 다른 샘플이라고 판정하지 않습니다.</p></section>
    <section><h2>Energy SLO Verification · TC4</h2><p>전력 원본은 두 장비의 tegrastats VDD_IN 순간값입니다. <b>서버의 train push_end → 모든 학습 응답 수신</b> 구간을 두 장비에 공통 적용하여 사다리꼴 적분했습니다. 학습·해당 구간 통신·먼저 끝난 장비의 대기는 포함하고, 평가·aggregation·냉각 대기는 제외합니다. 현재 onsite whole-round 경계와 동일한 측정으로 간주하지 않습니다.</p><p><b>서버–장비 시계 오차 검증이 없어 아래 에너지는 참고값입니다.</b> 3초 초과 전력 공백이나 경계 미포함이면 적분을 보류합니다. 원본 CSV의 energy=0은 전력 모니터 비활성의 기록이며, 아래 값은 별도 guard 로그에서 계산했습니다.</p>{etable}<p>사전에 확정한 slo_lock.json을 확인하지 못해 SLO 상한·위반률·TC4 PASS/FAIL을 표시하지 않습니다. R1–2로 사후 상한을 생성해 사전 기준처럼 사용하지 않았습니다.</p></section>
    <section><h2>다운로드 및 원본 증빙</h2><p><a href="nano_normal6_report.zip" download>전체 ZIP</a> · <a href="sample_workload_prediction_energy.csv" download>예측·실측·워크로드·에너지 CSV</a> · <a href="accuracy_summary.csv">정확도 요약</a> · <a href="tta_energy_audit.csv">에너지 상세</a> · <a href="power_samples.csv">전력 샘플</a> · <a href="tta_dashboard_data.json">화면 데이터 JSON</a> · <a href="checksums.sha256">SHA-256</a></p><p>ZIP에는 이 HTML, 모든 CSV/JSON, 원본 telemetry·의사결정·학습·서버 경계 로그 및 재생성 코드가 포함됩니다. 압축을 풀고 index.html을 열면 서버 설치 없이 볼 수 있습니다.</p></section><footer>표시된 PASS는 TC1 수치 기준입니다. 사전 SLO·시계 정렬·샘플 독립성 검증 없이 TTA 종합 PASS를 의미하지 않습니다.</footer>'''
    (OUT/'index.html').write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nano 정상 6라운드 결과</title><style>'+css+'</style></head><body><main class="wrap">'+body+'</main></body></html>')
    # Package all preserved sources; omit the archive itself from its checksum list.
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ('checksums.sha256','nano_normal6_report.zip'))
    (OUT/'checksums.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(OUT))+'\n' for p in files))
    with zipfile.ZipFile(OUT/'nano_normal6_report.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[OUT/'checksums.sha256']:z.write(p,'nano_normal6/'+str(p.relative_to(OUT)))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':build()
