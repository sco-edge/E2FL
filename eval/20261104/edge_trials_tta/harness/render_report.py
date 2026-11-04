#!/usr/bin/env python3
"""Loopback-only live view and portable final page for a TTA attempt."""
from __future__ import annotations
import argparse
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socketserver
import time


def load_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def load_csv(path):
    if not path.is_file(): return []
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def build(run):
    run=Path(run).resolve()
    state=load_json(run/'state.json') or {}
    summary=load_json(run/'report/tta_summary.json')
    config=load_json(run/'effective_config.json') or {}
    lock=load_json(run/'reference/slo_lock.json') or load_json(run/'slo_lock.json')
    inputs=load_csv(run/'report/tta_prediction_inputs.csv')
    prediction=load_csv(run/'report/tta_prediction_audit.csv')
    energy=load_csv(run/'report/tta_energy_audit.csv')
    dataset=load_json(run/'report/tta_dataset_independence.json')
    slo_audit=load_json(run/'report/tta_slo_audit.json')
    progress=load_json(run/'progress.json') or {}
    spec=run/'suite.toml'
    current_metric=bool(summary and summary.get('criteria',{}).get('accuracy_metric')=='within_error_limit'
                        and summary.get('audit_status')=='verified'
                        and summary.get('prediction_audit_valid') is True
                        and summary.get('slo_audit_valid') is True
                        and summary.get('dataset_independence_pass') is not None
                        and prediction and energy and slo_audit and dataset)
    status=('Pending' if not state or state.get('status') in ('starting','running','executed') else
            'SMOKE COMPLETE' if state.get('stage')=='smoke' and state.get('status')=='validated' else
            'CALIBRATED' if state.get('stage')=='calibration' and state.get('status')=='validated' else
            'PRIMED' if state.get('stage')=='prime' and state.get('status')=='validated' else
            'INVALID' if not current_metric else
            'PASS' if summary.get('pass') and state.get('status')=='complete' else
            'FAIL' if summary and state.get('status')=='complete' else
            'INVALID')
    return {
        'schema':1,'generated_at':time.time(),'run_name':run.name,
        'status':status,'stage':state.get('stage','Pending'),
        'state_status':state.get('status','Pending'),
        'progress_rounds':progress.get('rounds_predicted',[]),
        'rounds_total':config.get('num-server-rounds'),
        'test_name':'LoRACast TTA Onsite v2','test_cases':['TC1','TC4'],
        'devices':['RPi5','jetson_orin_nx'],
        'model':config.get('model','Pending'),'dataset':config.get('dataset','Pending'),
        'config_path':str(spec) if spec.is_file() else None,
        'config_sha256':hashlib.sha256(spec.read_bytes()).hexdigest() if spec.is_file() else None,
        'git_commit':(load_json(run/'source_commit.json') or {}).get('commit'),
        'config':config,'prediction_inputs':inputs,'prediction_audit':prediction,
        'energy_audit':energy,'dataset_independence':dataset,
        'slo_audit':slo_audit,'slo_lock':lock,'summary':summary,
        'measurement_methods':{'RPi5':'PMIC matched rail sum',
                               'jetson_orin_nx':'INA3221 VDD_IN'},
        'note':'PASS/FAIL은 analyze.py의 보존된 결과만 표시합니다. 미측정 값은 Pending입니다.',
    }


HTML=r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LoRACast TTA Onsite v2</title>
<style>
:root{--bg:#f4f7fb;--ink:#17283e;--muted:#66768b;--line:#d9e3ee;--blue:#235993;--ok:#087d56;--bad:#b6443e}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}.wrap{max-width:1260px;margin:auto;padding:24px}header{background:linear-gradient(120deg,#142b4c,#266093);border-radius:18px;color:#fff;padding:28px 34px}h1{margin:4px 0;font-size:30px}h2{margin:0 0 14px;font-size:21px}.sub{opacity:.85}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.card,section{background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 3px 15px #183e5b0b}.card small{display:block;color:var(--muted)}.card strong{display:block;font-size:25px;margin:5px 0}.card small:last-child{font-size:12px}.PASS{color:var(--ok)}.FAIL,.INVALID{color:var(--bad)}.Pending{color:var(--muted)}section{margin:14px 0}table{border-collapse:collapse;width:100%;min-width:820px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#f7f9fc;color:var(--muted);font-size:12px}td{font-variant-numeric:tabular-nums}.scroll{overflow:auto}.hint{color:var(--muted);font-size:13px}.tag{display:inline-block;border-radius:999px;background:#e8eff7;color:#21466c;padding:2px 8px;font-size:12px}.bars{display:grid;gap:8px}.barrow{display:grid;grid-template-columns:120px 1fr 90px;gap:8px;align-items:center}.track{height:11px;border-radius:8px;background:#e6edf5;overflow:hidden}.fill{height:100%;background:#2d6da6}.fill.actual{background:#e49b34}.mono{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px;overflow-wrap:anywhere}footer{color:var(--muted);font-size:12px;padding:12px 2px}@media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}.barrow{grid-template-columns:85px 1fr 70px}}@media(max-width:550px){.wrap{padding:10px}.grid{gap:8px}.card{padding:12px}header{padding:22px}.card strong{font-size:19px}}
</style></head><body><main class="wrap"><header><div>TC1 · TC4 · 실시간 시험 증빙</div><h1>LoRACast TTA Onsite v2</h1><div id="head" class="sub">Loading</div></header><div class="grid" id="cards"></div><section><h2>Test Overview</h2><div id="overview"></div></section><section><h2>Workload Definitions</h2><p class="hint">Sample Cap은 실제 처리 샘플 수와 다릅니다. Local Steps=0은 step 상한 없음입니다. 시험용 client tokenizer는 설정된 Sequence Length Cap을 실제 절단에 적용합니다. 각 샘플의 관측 토큰 수와 학습 토큰 합계는 audit에서 확인합니다.</p><div class="scroll" id="definitions"></div></section><section><h2>Prediction Verification · TC1</h2><p class="hint">초기 보정 1–2라운드와 평가 라운드를 분리합니다. 상태 재사용 모드에서는 보정이 별도 prime 실행입니다. 주 지표는 상대오차 ≤10%인 유효 표본의 비율이며, 6표본이면 6개 모두 성공해야 90%에 도달합니다.</p><div id="predbars" class="bars"></div><div class="scroll" id="predtable"></div><p id="dataset" class="hint"></p></section><section><h2>Energy SLO Verification · TC4</h2><p class="hint">Jetson INA3221 VDD_IN, Pi 5 PMIC 합산값. 로컬 학습·통신·idle waiting을 포함한 라운드 전체 적분입니다. SLO = 보정 5라운드 최대 J × 1.10이며 TTA 지정 공식 기준 여부는 별도 협의 대상입니다.</p><div id="slo"></div><div class="scroll" id="energytable"></div></section><footer id="footer"></footer></main>
<script>const EMBEDDED=__DATA__;const esc=x=>String(x??'Pending').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const n=(x,d=2)=>x==null||x===''?'Pending':Number(x).toFixed(d);const ts=x=>{if(x==null||x==='')return'Pending';const t=Number.isFinite(Number(x))?new Date(Number(x)*1000):new Date(x);return Number.isNaN(t.getTime())?'Pending':t.toLocaleString()};const set=(id,v)=>document.getElementById(id).innerHTML=v;
const defs=[['Device','i','기기 종류와 식별자','식별자','예측 입력','device_name / device_uid'],['Sample Cap','S','라운드 최대 선택 샘플','개','예측 입력','max_samples_per_round'],['Selected Samples','—','실제 선택된 샘플 수','개','관측값','selected_samples'],['Estimated Tokens / Example','평균 길이','기본 모드는 이전 라운드 관측 평균; 진단 모드는 사전 고정 길이','token/example','예측 입력','estimated_tokens_per_example'],['Local Steps','H','optimizer update step 상한; 0이면 상한 없음','step','조건부 예측 입력','local_steps'],['Local Epochs','E','로컬 데이터 반복 횟수','회','예측 입력','local_epochs'],['Batch Size','B','한 단계의 샘플 수','개','step cap 사용 시 간접 입력','batch_size'],['Sequence Length Setting','L','현재 시험용 토크나이저의 실제 입력 상한','token','학습환경 + history 없는 fallback에서 간접 입력','max_seq_len'],['LoRA Rank','ρ','저랭크 행렬 차원','차원','학습환경','lora_rank'],['LoRA Modules','M','LoRA 적용 계층','계층','학습환경','lora_modules'],['Model','—','미세조정 언어모델','—','학습환경','model'],['Dataset','—','로컬 학습 데이터셋','—','학습환경','dataset']];
function render(d){const s=d.summary,o=s?.overall,lock=d.slo_lock,rows=d.prediction_audit||[],inputs=d.prediction_inputs||[],erows=d.energy_audit||[];set('head',esc(d.run_name)+' · '+esc(d.stage)+' · '+esc(d.state_status));set('cards',[["종합 판정",d.status,"분석기 결과만 표시"],["진행",(d.progress_rounds?.length||0)+' / '+(d.rounds_total??'Pending')+' rounds',"예측 생성 기준"],["TC1 정확도",o?n(o.time_prediction_accuracy_pct)+'%':'Pending','목표 ≥90% · 보조 (1−MARE) '+(o?n(o.one_minus_mare_accuracy_pct)+'%':'Pending')],["TC4 SLO 위반",o?n(o.energy_violation_rate_pct)+'%':'Pending','목표 ≤10%']].map(x=>'<div class="card"><small>'+esc(x[0])+'</small><strong class="'+esc(x[1])+'">'+esc(x[1])+'</strong><small>'+esc(x[2])+'</small></div>').join(''));
set('overview','<p><b>기기</b> '+esc(d.devices.join(' / '))+'　<b>모델</b> '+esc(d.model)+'　<b>데이터셋</b> '+esc(d.dataset)+'</p><p><b>설정 파일</b> <span class="mono">'+esc(d.config_path)+'</span><br><b>설정 SHA-256</b> <span class="mono">'+esc(d.config_sha256)+'</span><br><b>Git commit</b> <span class="mono">'+esc(d.git_commit)+'</span></p>');
const byDev={};for(const x of inputs){(byDev[x.device_name]??={})[x.round]=x}const latest=dev=>Object.values(byDev[dev]||{}).sort((a,b)=>Number(b.round)-Number(a.round))[0]||{};const configured={max_samples_per_round:d.config['max-samples-per-round'],local_steps:0,local_epochs:d.config['local-epochs'],batch_size:d.config.batch_size,max_seq_len:d.config['max-seq-len'],lora_rank:d.config['lora-r'],lora_modules:d.config['lora-target-modules'],model:d.model,dataset:d.dataset};const value=(dev,key)=>{const v=latest(dev)[key];return v!==null&&v!==undefined&&v!==''?v:(configured[key]??'Pending')};set('definitions','<table><thead><tr><th>항목</th><th>기호</th><th>정의</th><th>단위</th><th>역할</th><th>Pi 5</th><th>Jetson</th></tr></thead><tbody>'+defs.map(([name,sym,desc,unit,role,key])=>'<tr><td>'+esc(name)+'</td><td>'+esc(sym)+'</td><td>'+esc(desc)+'</td><td>'+esc(unit)+'</td><td>'+esc(role)+'</td><td>'+esc(value('RPi5',key))+'</td><td>'+esc(value('jetson_orin_nx',key))+'</td></tr>').join('')+'</tbody></table>');
const maxT=Math.max(1,...rows.flatMap(x=>[Number(x.pred_train_time_self)||0,Number(x.actual_train_time_self)||0]));set('predbars',rows.map(x=>{let label='R'+esc(x.logical_round||x.round)+' '+(x.device_name==='RPi5'?'Pi':'Jetson');return '<div class="barrow"><span>'+label+'</span><div><div class="track"><div class="fill" style="width:'+Math.min(100,100*Number(x.pred_train_time_self)/maxT)+'%"></div></div><div class="track" style="margin-top:3px"><div class="fill actual" style="width:'+Math.min(100,100*Number(x.actual_train_time_self)/maxT)+'%"></div></div></div><span>'+n(x.pred_train_time_self,1)+' / '+n(x.actual_train_time_self,1)+'</span></div>'}).join('')+'<p class="hint">파랑 예측 · 주황 실제 (초)</p>');set('predtable','<table><thead><tr><th>논리 / Flower Round</th><th>Device / UID</th><th>S / selected / H / E / B / L설정</th><th>LoRA r / modules</th><th>평균 토큰수 · 출처</th><th>β / α · 상태 갱신</th><th>예측 시각 · 학습 시작/종료</th><th>예측 / 실제 / 절대오차</th><th>상대오차 · 성공</th><th>감사</th></tr></thead><tbody>'+rows.map(x=>{const i=inputs.find(y=>y.round===x.round&&y.device_name===x.device_name)||{};const valid=x.valid_prediction==='True';const success=valid&&x.relative_error!==''&&Number(x.relative_error)<=Number(s?.criteria?.time_error_limit??0.1);return '<tr><td>'+esc(x.logical_round||x.round)+' / '+esc(x.round)+'</td><td>'+esc(x.device_name)+'<br>'+esc(x.device_uid)+'</td><td>'+[i.max_samples_per_round,i.selected_samples,i.local_steps,i.local_epochs,i.batch_size,i.max_seq_len].map(esc).join(' / ')+'</td><td>'+esc(i.lora_rank)+' / '+esc(i.lora_modules)+'</td><td>'+n(i.estimated_tokens_per_example,1)+' / '+esc(i.token_workload_source)+'</td><td>'+n(i.beta_sec_per_token,5)+' / '+n(i.alpha,3)+'<br>last '+esc(i.coefficient_last_updated_round)+' · history '+esc(x.history_rounds)+'</td><td>'+ts(x.prediction_timestamp)+'<br>'+ts(x.training_start_timestamp)+'<br>'+ts(x.training_end_timestamp)+'</td><td>'+n(x.pred_train_time_self)+' / '+n(x.actual_train_time_self)+' / '+n(x.absolute_error_s)+' s</td><td>'+n(x.relative_error===''?null:Number(x.relative_error)*100)+'%<br>'+esc(valid?(success?'성공':'실패'):'무효')+'</td><td>'+esc(valid?'유효':x.invalid_reasons)+'</td></tr>'}).join('')+'</tbody></table>');const counts=d.dataset_independence?.observed_input_token_counts;set('dataset','학습 샘플 독립성: '+esc(d.dataset_independence?.status)+' · 교집합 '+esc(d.dataset_independence?.overlap_count)+'개 · 관측 입력 토큰 수 Pi '+esc(counts?.RPi5?.evaluation?.join(', '))+' / Jetson '+esc(counts?.jetson_orin_nx?.evaluation?.join(', '))+'. 중복이 있으면 TC1 독립성 요건을 충족하지 않습니다.');
const cal=lock?.calibration_energy_j||{};set('slo','<p><b>사전 잠금:</b> '+esc(d.slo_audit?.unchanged===true?'변경 없음':'Pending / 확인 필요')+'　<b>잠금 시각:</b> '+ts(lock?.created_at)+'　<b>보정:</b> '+esc(lock?.measurement_rounds)+' rounds　<b>margin:</b> '+esc(lock?.margin)+'</p><p>Pi 5 SLO '+n(lock?.slo_joules?.RPi5)+' J · 보정 '+esc(cal.RPi5?.map(x=>n(x,1)).join(', '))+' J<br>Jetson SLO '+n(lock?.slo_joules?.jetson_orin_nx)+' J · 보정 '+esc(cal.jetson_orin_nx?.map(x=>n(x,1)).join(', '))+' J</p>');set('energytable','<table><thead><tr><th>논리 / Flower Round</th><th>Device</th><th>시작 / 종료</th><th>Duration</th><th>Power samples</th><th>최대 간격</th><th>에너지 / SLO</th><th>위반</th><th>경계 검증</th></tr></thead><tbody>'+erows.map(x=>'<tr><td>'+esc(x.logical_round||x.round)+' / '+esc(x.round)+'</td><td>'+esc(x.device_name)+'</td><td>'+ts(x.round_start_timestamp)+'<br>'+ts(x.round_end_timestamp)+'</td><td>'+n(x.round_duration_s)+' s</td><td>'+esc(x.power_samples)+'</td><td>'+n(x.max_power_sample_gap_s)+' s</td><td>'+n(x.round_energy_j)+' / '+n(x.energy_slo_j)+' J</td><td>'+esc(x.energy_violation)+'</td><td>'+esc(x.valid_energy==='True'?'유효':'확인 필요')+'</td></tr>').join('')+'</tbody></table>');set('footer',esc(d.note)+' · 데이터 갱신 '+new Date(d.generated_at*1000).toLocaleString());}
render(EMBEDDED);if(location.protocol==='http:'||location.protocol==='https:'){setInterval(async()=>{try{const r=await fetch('/api/data',{cache:'no-store'});if(r.ok)render(await r.json())}catch(e){}},5000)}
</script></body></html>'''


def html(data):
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    return HTML.replace('__DATA__',payload)


def export(run):
    run=Path(run).resolve(); report=run/'report'; report.mkdir(exist_ok=True)
    data=build(run)
    (report/'tta_dashboard_data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
    (report/'render_report.html').write_text(html(data))
    return report/'render_report.html'


def serve(run,port):
    class LocalServer(ThreadingHTTPServer):
        def server_bind(self):
            # Avoid a reverse-DNS lookup while opening a loopback dashboard.
            socketserver.TCPServer.server_bind(self)
            self.server_name='127.0.0.1'
            self.server_port=self.server_address[1]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ('/','/api/data'):
                self.send_error(404); return
            body=(json.dumps(build(run),ensure_ascii=False,allow_nan=False).encode()
                  if self.path=='/api/data' else html(build(run)).encode())
            self.send_response(200); self.send_header('Content-Type','application/json; charset=utf-8' if self.path=='/api/data' else 'text/html; charset=utf-8')
            self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    with LocalServer(('127.0.0.1',port),Handler) as server:
        print(f'http://127.0.0.1:{server.server_port}/',flush=True); server.serve_forever()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--serve',action='store_true'); parser.add_argument('--port',type=int,default=8765)
    a=parser.parse_args()
    if a.serve: serve(a.run_dir,a.port)
    else: print(export(a.run_dir))


if __name__=='__main__': main()
