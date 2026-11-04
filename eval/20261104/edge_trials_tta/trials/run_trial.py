#!/usr/bin/env python3
"""Run the frozen TTA harness and immediately show a portable result page.

This file lives outside both fingerprinted harnesses and never modifies their
evidence directories or SLO locks.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from export_workload import export as export_workload_csv

ROOT = Path(__file__).resolve().parents[1]
HARNESSES = {
    'reuse': (ROOT / 'harness', 'suite.toml'),
    'standard': (ROOT / 'tta_harness', 'tta_suite_quota2.toml'),
}
RESULTS = Path(__file__).resolve().parent / 'results'


def load(path):
    return json.loads(Path(path).read_text())


def h(value):
    return html.escape(str(value), quote=True)


def number(value, decimals=1):
    return f'{float(value):,.{decimals}f}'


def duration(seconds):
    seconds = round(float(seconds))
    return f'{seconds // 60}분 {seconds % 60:02d}초'


def integrity(run):
    manifest = run / 'checksums.json'
    if not manifest.is_file():
        return False, '증거 파일 목록이 없습니다.'
    try:
        hashes = load(manifest)
        if not isinstance(hashes, dict) or not hashes:
            return False, '증거 파일 목록이 비어 있습니다.'
        for name, expected in hashes.items():
            path = (run / name).resolve()
            if not path.is_relative_to(run.resolve()) or not path.is_file():
                return False, f'증거 파일 누락: {name}'
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return False, f'증거 파일 변경: {name}'
    except (OSError, ValueError, TypeError) as exc:
        return False, f'증거 확인 오류: {exc}'
    return True, f'{len(hashes)}개 파일 해시 일치'


def calibration_context(run, summary):
    """Read the frozen reference, never infer a limit from onsite observations."""
    path = run / 'reference/slo_lock.json'
    if not path.is_file():
        return None, '보정 기준 파일이 없습니다.'
    if hashlib.sha256(path.read_bytes()).hexdigest() != summary.get('slo_lock_sha256'):
        return None, '보정 기준 해시가 결과 기록과 다릅니다.'
    try:
        lock = load(path)
        margin = float(lock['margin'])
        count = int(lock['measurement_rounds'])
        if not math.isfinite(margin) or margin < 1 or count < 1:
            raise ValueError('invalid margin or count')
        maxima = {}
        for name, limit in summary['slo_joules'].items():
            values = [float(x) for x in lock['calibration_energy_j'][name]]
            if len(values) != count or any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError(f'invalid calibration observations: {name}')
            maximum = max(values)
            if not math.isclose(maximum * margin, float(limit), rel_tol=1e-8):
                raise ValueError(f'SLO formula mismatch: {name}')
            maxima[name] = maximum
        if int(lock['warmup_rounds']) != int(summary['criteria']['warmup_rounds_excluded']):
            raise ValueError('calibration and onsite warm-up definitions differ')
    except (KeyError, ValueError, TypeError, OverflowError) as exc:
        return None, f'보정 기준 내용이 일치하지 않습니다: {exc}'
    return {'margin': margin, 'count': count, 'maxima': maxima,
            'merged': lock.get('slo_source') == 'prime',
            'created_at': lock.get('created_at', '기록 없음')}, None


def criteria_text(run):
    """Short supplement suitable for sharing with the proposed test report."""
    run = Path(run).resolve()
    summary_path = run / 'report/tta_summary.json'
    if not summary_path.is_file():
        return '에너지 상한 산정 근거: 유효한 현장시험 결과가 없어 작성할 수 없습니다.\n'
    summary = load(summary_path)
    context, error = calibration_context(run, summary)
    if context is None:
        return f'에너지 상한 산정 근거: 확인 불가. {error}\n'
    lines = [
        'LoRACast 에너지 상한 산정 근거 (제안 시험 기준)',
        '',
        f"같은 시험 조건에서 사전에 측정한 {context['count']}라운드의 기기별 최대 라운드 에너지에 "
        f"{number((context['margin'] - 1) * 100, 0)}% 여유를 적용했습니다. "
        '상한은 현장시험 전에 잠금 파일로 고정하며 사후 조정하지 않습니다.',
        f"보정 잠금 시각: {context['created_at']}",
        '산식: 기기별 상한(J) = 사전 보정 라운드 에너지 최댓값(J) × '
        f"{number(context['margin'], 2)}",
    ]
    for name, label in (('RPi5', 'Pi 5'), ('jetson_orin_nx', 'Jetson Orin NX')):
        if name in context['maxima']:
            lines.append(f"{label}: {number(context['maxima'][name], 2)} J × "
                         f"{number(context['margin'], 2)} = "
                         f"{number(summary['slo_joules'][name], 2)} J")
    lines += [
        '',
        'TC1 주 지표: 상대오차 |예측-실제|/실제 ≤ 10%인 유효 표본 수 ÷ 전체 유효 표본 수 × 100. 목표 ≥ 90%.',
        '기존 (1−MARE)×100은 보조 지표이며 판정에는 사용하지 않습니다.',
        ('학습 샘플 독립성은 prime/SLO 보정과 official의 실제 읽기 해시 교집합으로 판정합니다.'
         if context['merged'] else
         '학습 샘플 독립성은 prime·SLO 보정·official의 실제 읽기 해시를 세 쌍 모두 비교해 판정합니다.'),
        '',
        '해석 범위: 동일 장치·워크로드·센서·측정 구간에서 에너지 사용이 사전 설정한 경험적 범위에 머무는지 확인합니다.',
        '한계: 자체 보정 최댓값과 선택한 여유율에 기반하므로 절대적인 에너지 효율, 다른 방식 대비 절감 효과, 기기 간 공정성을 입증하지 않습니다.',
        f"보정 {context['count']}회와 현장 측정 3회만으로 장기 위반률을 일반화할 수 없습니다. Pi PMIC 합산값과 Jetson VDD_IN 값의 절대량도 직접 비교하지 않습니다.",
        '이 상한과 여유율은 기관이 지정한 공식 기준이 아닌 제안 시험 조건입니다.',
        f"보정 기준 SHA-256: {summary['slo_lock_sha256']}",
        '',
    ]
    return '\n'.join(lines)


def chip(label, good=None):
    klass = 'neutral' if good is None else 'good' if good else 'bad'
    return f'<span class="chip {klass}">{h(label)}</span>'


def card(label, value, detail='', good=None):
    klass = 'good' if good is True else 'bad' if good is False else ''
    return (f'<div class="metric {klass}"><div class="metric-label">{h(label)}</div>'
            f'<div class="metric-value">{h(value)}</div><div class="metric-detail">{h(detail)}</div></div>')


def page(run):
    run = Path(run).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    state_path = run / 'state.json'
    state = load(state_path) if state_path.is_file() else {}
    ok_hash, hash_detail = integrity(run)
    summary_path = run / 'report/tta_summary.json'
    summary = load(summary_path) if summary_path.is_file() and (run / 'report/tta_observations.csv').is_file() else None
    diagnosis_path = run / 'diagnosis.json'
    diagnosis = load(diagnosis_path) if diagnosis_path.is_file() else {}
    date = datetime.fromtimestamp(state.get('start', run.stat().st_mtime)).astimezone().strftime('%Y-%m-%d %H:%M')
    title = 'LoRACast 현장시험 결과'
    if summary:
        complete = bool(summary.get('overall', {}).get('complete'))
        calibration, calibration_error = calibration_context(run, summary)
        evidence_valid = (ok_hash and state.get('status') == 'complete' and calibration is not None
                          and summary.get('criteria',{}).get('accuracy_metric') == 'within_error_limit'
                          and summary.get('audit_status') == 'verified'
                          and summary.get('prediction_audit_valid') is True
                          and summary.get('slo_audit_valid') is True
                          and summary.get('dataset_independence_pass') is not None)
        passed = bool(summary.get('pass')) and complete and evidence_valid
        target = summary.get('onsite_runtime', {})
        within = target.get('within_target') is True
        elapsed = target.get('elapsed_s')
        observations_path = run / 'report/tta_observations.csv'
        with observations_path.open(newline='', encoding='utf-8-sig') as f:
            observations = list(csv.DictReader(f))
        criteria = summary['criteria']
        warmup = int(criteria.get('warmup_rounds_excluded', 0))
        measured = int(criteria.get('official_measurement_rounds_per_device', 3))
        per_device = summary['per_device']
        status = 'INVALID' if not evidence_valid or not complete else 'PASS' if passed else 'FAIL'
        format_label = f'준비 {warmup} + 측정 {measured}라운드' if warmup else f'측정 {measured}라운드'
        eyebrow = f'제안 시험 기준 · {h(format_label)} · {h(run.name)} · {h(date)}'
        headline = ('결과 확인 필요' if status == 'INVALID' else
                    f'{measured}라운드 측정 통과' if passed else '시험 기준 미충족')
        subtitle = ('증거 파일이나 실행 완료 상태를 확인해야 합니다.' if status == 'INVALID' else
                    'Pi와 Jetson의 모든 측정값을 확인했습니다.')
        top = (card('종합 판정', status, '시간 예측 + 에너지 SLO', passed)
               + card('현장 소요 시간', duration(elapsed) if elapsed is not None else '확인 불가',
                      f"목표 {duration(target.get('target_s', 600))} · 사전 점검 포함", within)
               + card('시간 예측 정확도', f"{number(summary['overall']['time_prediction_accuracy_pct'], 2)}%",
                      f"기준 {number(float(criteria['time_accuracy_target'])*100, 0)}% 이상",
                      summary['overall']['time_pass'])
               + card('에너지 SLO 위반',
                      f"{sum(str(row['energy_violation']).lower() == 'true' for row in observations)} / {len(observations)}회",
                      f"허용률 {number(float(criteria['energy_violation_target'])*100, 0)}% 이하",
                      summary['overall']['energy_pass']))
        devices = []
        for name, label in (('RPi5', 'Raspberry Pi 5'), ('jetson_orin_nx', 'Jetson Orin NX')):
            if name not in per_device:
                continue
            item = per_device[name]
            budget = summary['slo_joules'][name]
            devices.append('<div class="device"><div class="device-head"><strong>' + h(label) + '</strong>'
                           + chip('PASS' if item['time_pass'] and item['energy_pass'] else 'FAIL',
                                  item['time_pass'] and item['energy_pass']) + '</div>'
                           + '<div class="device-numbers"><div><small>예측 정확도</small><b>'
                           + h(number(item['time_prediction_accuracy_pct'], 2)) + '%</b></div>'
                           + '<div><small>에너지 상한</small><b>' + h(number(budget, 1)) + ' J</b></div>'
                           + '<div><small>유효 측정</small><b>' + h(item['observations']) + ' / '
                           + h(item['expected_observations']) + '</b></div></div></div>')
        table_rows = []
        for row in sorted(observations, key=lambda x: (int(x['round']), x['device_name'])):
            name = 'Pi 5' if row['device_name'] == 'RPi5' else 'Jetson' if row['device_name'] == 'jetson_orin_nx' else row['device_name']
            err = float(row['relative_error']) * 100
            energy = float(row['round_energy_j'])
            budget = float(row['energy_slo_j'])
            time_within = str(row['time_pass']).lower() == 'true'
            energy_within = str(row['energy_violation']).lower() == 'false'
            if not energy_within:
                row_label, row_good = '에너지 상한 초과', False
            elif not time_within:
                row_label, row_good = '시간 오차 10% 초과', None
            else:
                row_label, row_good = '개별 기준 이내', True
            table_rows.append('<tr><td>' + h(row['round']) + '</td><td>' + h(name) + '</td>'
                              + '<td>' + h(number(row['pred_train_time_s'], 1)) + '초</td>'
                              + '<td>' + h(number(row['actual_train_time_s'], 1)) + '초</td>'
                              + '<td>' + h(number(err, 1)) + '%</td>'
                              + '<td>' + h(number(energy, 1)) + ' / ' + h(number(budget, 1)) + ' J</td>'
                              + '<td>' + chip(row_label, row_good) + '</td></tr>')
        details = ('<section class="section"><div class="section-title"><h2>기기별 결과</h2><p>각 기기에서 ' + h(measured) + '회씩 측정</p></div>'
                   + '<div class="devices">' + ''.join(devices) + '</div></section>'
                   + '<section class="section"><div class="section-title"><h2>라운드별 측정</h2><p>학습 시간 예측과 실제값, 라운드 전체 에너지</p></div>'
                   + '<p class="explain">최종 시간 판정은 전체 유효 표본 중 상대오차 10% 이하인 표본의 비율이 90% 이상인지로 결정합니다. (1−MARE)는 보조 지표입니다.</p>'
                   + '<div class="table-scroll"><table><thead><tr><th>라운드</th><th>기기</th><th>예측</th><th>실제</th><th>오차</th><th>에너지 / 상한</th><th>개별 점검</th></tr></thead>'
                   + '<tbody>' + ''.join(table_rows) + '</tbody></table></div></section>')
        dataset_path = run / 'report/tta_dataset_independence.json'
        dataset = load(dataset_path) if ok_hash and dataset_path.is_file() else None
        if dataset and dataset.get('schema',0) >= 2:
            pairwise = dataset.get('pairwise_overlap_counts',{})
            merged = 'slo_calibration' not in dataset.get('phase_sample_ids',{})
            comparisons = ([('prime/SLO 보정 vs official','prime_vs_official')] if merged else
                           [('prime vs SLO 보정','prime_vs_slo_calibration'),
                            ('prime vs official','prime_vs_official'),
                            ('SLO 보정 vs official','slo_calibration_vs_official')])
            split_rows = ''.join('<tr><td>'+h(label)+'</td><td>'
                                 +h(pairwise.get(key,'미측정'))+'개</td></tr>'
                                 for label,key in comparisons)
            details += ('<section class="section"><div class="section-title"><h2>학습 샘플 분리</h2>'
                        '<p>실제 읽힌 토큰 해시 기준</p></div><p>각 기기 partition의 '
                        + ('prime/SLO 보정 행 2–3, official 행 4–5' if merged else
                           'prime 행 0–1, SLO 보정 행 2–3, official 행 4–5')
                        + '를 사용했습니다. 각 단계 안에서는 '
                        '라운드마다 같은 행을 다시 읽습니다.</p><div class="table-scroll"><table>'
                        '<thead><tr><th>비교</th><th>교집합</th></tr></thead><tbody>'
                        +split_rows+'</tbody></table></div><p class="explain">감사 상태: '
                        +h(dataset.get('status','미측정'))
                        +' · 원시 샘플 목록: report/tta_dataset_samples.csv</p></section>')
        if calibration:
            calibration_rows = []
            for name, label in (('RPi5', 'Pi 5'), ('jetson_orin_nx', 'Jetson Orin NX')):
                if name in calibration['maxima']:
                    calibration_rows.append('<tr><td>' + h(label) + '</td><td>'
                                            + h(number(calibration['maxima'][name], 2)) + ' J</td><td>× '
                                            + h(number(calibration['margin'], 2)) + '</td><td><strong>'
                                            + h(number(summary['slo_joules'][name], 2)) + ' J</strong></td></tr>')
            rationale = ('<p>같은 조건에서 <strong>현장시험 전에</strong> 측정한 '
                         + h(calibration['count']) + '라운드의 기기별 최대 에너지에 '
                         + h(number((calibration['margin'] - 1) * 100, 0))
                         + '% 여유를 적용하고 기준을 잠갔습니다. 현장 결과를 보고 상한을 바꾸지 않습니다.</p>'
                         + '<p class="explain">보정 잠금 시각: ' + h(calibration['created_at']) + '</p>'
                         + '<div class="table-scroll"><table><thead><tr><th>기기</th><th>보정 최댓값</th><th>사전 여유율</th><th>고정 상한</th></tr></thead><tbody>'
                         + ''.join(calibration_rows) + '</tbody></table></div>')
        else:
            rationale = '<p class="failure">' + h(calibration_error) + '</p>'
        rationale += ('<p class="scope"><strong>해석 범위:</strong> 동일 장치·워크로드·센서·측정 구간에서 에너지가 사전 설정한 경험적 범위에 머무는지 확인하는 기준입니다. '
                      '10% 여유율은 시험자가 사전에 선택한 값이며 기관 지정 기준이 아닙니다. '
                      '자체 보정 최댓값에 기반하므로 절대적 에너지 효율이나 다른 방법 대비 절감 효과를 입증하지 않습니다. '
                      'Pi와 Jetson은 계측 범위가 달라 절대 J 값을 기기 간 공정성 비교에 쓰지 않습니다. '
                      '짧은 보정과 현장 측정만으로 장기 위반률을 일반화할 수 없습니다.</p>')
        details += ('<section class="section"><div class="section-title"><h2>에너지 상한을 정한 이유</h2><p>제안 시험 조건 · 사전 고정</p></div>'
                    + rationale + '</section>')
        meta = (chip('증거 확인: ' + hash_detail, ok_hash)
                + chip('보정 기준 확인' if calibration else '보정 기준 확인 불가', calibration is not None)
                + chip('Wi-Fi 직접 통신' if state.get('transport') == 'direct' else '통신 방식 확인 필요',
                       state.get('transport') == 'direct')
                + chip(f'{measured}라운드 모두 측정' if complete else '측정 누락', complete))
        notes = ('<details><summary>측정 기준과 파일</summary><p>이 결과는 사전에 정한 제안 시험 기준의 검증 결과이며, 기관의 인증 여부를 뜻하지 않습니다. '
                 + h(criteria['energy_definition']) + '</p><p>측정 자료: <code>' + h(run) + '</code></p>'
                 + '<p>SLO 잠금 해시: <code>' + h(summary.get('slo_lock_sha256', '')) + '</code></p></details>')
    else:
        status = 'INVALID'
        eyebrow = f'실행 점검 · {h(run.name)} · {h(date)}'
        headline = '유효한 시험 결과가 없습니다'
        subtitle = '실행이나 증거 수집이 끝나지 않아 PASS/FAIL을 판정할 수 없습니다.'
        elapsed_path = run / 'wall_time.json'
        elapsed = load(elapsed_path).get('elapsed_s') if elapsed_path.is_file() else None
        top = (card('실행 상태', state.get('status', '미확인').upper(), '이 실행은 시험 판정에서 제외', False)
               + card('경과 시간', duration(elapsed) if elapsed is not None else '확인 불가'))
        suggestions = diagnosis.get('suggestions', [])
        if not isinstance(suggestions, list):
            suggestions = [suggestions]
        details = ('<section class="section"><div class="section-title"><h2>확인할 내용</h2></div>'
                   + '<div class="failure"><strong>' + h(diagnosis.get('error', state.get('error', '원인 기록이 없습니다.')))
                   + '</strong><ul>' + ''.join('<li>' + h(item) + '</li>' for item in suggestions)
                   + '</ul></div></section>')
        meta = chip('증거 확인: ' + hash_detail, ok_hash)
        notes = '<details><summary>실행 파일 위치</summary><p><code>' + h(run) + '</code></p></details>'
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{h(title)} — {h(status)}</title>
<style>
:root{{--ink:#12233a;--muted:#617187;--line:#dbe4ee;--blue:#184a89;--ok:#157a54;--bad:#aa3d39;--paper:#f3f7fb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}}
.wrap{{max-width:1120px;margin:auto;padding:32px 24px 56px}} header{{background:linear-gradient(120deg,#102d55,#23568a);color:#fff;border-radius:20px;padding:34px 40px;box-shadow:0 12px 35px #183b6326}}
.eyebrow{{font-size:13px;letter-spacing:.05em;opacity:.78;font-weight:700}} h1{{font-size:38px;line-height:1.15;margin:12px 0 8px}} .subtitle{{margin:0;opacity:.87}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}} .metric,.section{{background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:0 4px 20px #193e6810}}
.metric{{padding:20px 22px;min-height:132px;border-top:4px solid var(--blue)}} .metric.good{{border-top-color:var(--ok)}} .metric.bad{{border-top-color:var(--bad)}} .metric-label,small{{color:var(--muted);font-size:13px;font-weight:700}} .metric-value{{font-size:29px;font-weight:800;letter-spacing:-.04em;margin:5px 0}} .metric-detail{{color:var(--muted);font-size:13px}}
.section{{padding:24px 28px;margin-top:16px}} .section-title{{display:flex;justify-content:space-between;gap:12px;align-items:baseline;margin-bottom:16px}} h2{{font-size:21px;margin:0}} .section-title p{{margin:0;color:var(--muted);font-size:14px}}
.explain{{margin:0 0 12px;color:var(--muted);font-size:13px}}
.scope{{margin:16px 0 0;color:var(--muted);font-size:14px}}
.devices{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} .device{{border:1px solid var(--line);border-radius:13px;padding:18px 20px}} .device-head{{display:flex;justify-content:space-between;align-items:center;font-size:18px}} .device-numbers{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:16px}} .device-numbers div{{display:flex;flex-direction:column}} .device-numbers b{{font-size:18px}}
.chip{{display:inline-block;border-radius:999px;padding:4px 10px;font-size:12px;font-weight:800;white-space:nowrap}} .chip.good{{background:#dff4e9;color:#0c6845}} .chip.bad{{background:#fce8e7;color:#98312d}} .chip.neutral{{background:#e8eef6;color:#315474}}
.table-scroll{{overflow-x:auto}} table{{width:100%;border-collapse:collapse;text-align:left;white-space:nowrap}} th{{color:var(--muted);font-size:13px;background:#f7f9fc}} th,td{{padding:11px 12px;border-bottom:1px solid var(--line)}} tbody tr:last-child td{{border-bottom:0}} td{{font-variant-numeric:tabular-nums}} .meta{{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}} details{{color:var(--muted);font-size:13px}} summary{{cursor:pointer;font-weight:700}} code{{overflow-wrap:anywhere}} .failure{{background:#fff5f4;border:1px solid #f4ceca;border-radius:12px;padding:18px 20px}} .failure strong{{white-space:pre-wrap;overflow-wrap:anywhere}} .failure li{{margin-top:8px}}
@media(max-width:850px){{.grid{{grid-template-columns:repeat(2,1fr)}}.devices{{grid-template-columns:1fr}}}} @media(max-width:560px){{.wrap{{padding:12px 12px 28px}}header{{padding:26px 22px}}h1{{font-size:30px}}.grid{{gap:8px}}.metric{{padding:15px;min-height:120px}}.metric-value{{font-size:22px}}.section{{padding:18px 14px}}.section-title{{display:block}}.device-numbers b{{font-size:16px}}}}
</style></head><body><main class="wrap"><header><div class="eyebrow">{eyebrow}</div><h1>{h(headline)}</h1><p class="subtitle">{h(subtitle)}</p></header>
<div class="grid">{top}</div>{details}<div class="meta">{meta}</div>{notes}</main></body></html>'''


def publish(run, output_dir=RESULTS):
    run = Path(run).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f'{run.name}.html'
    csv_dest = output_dir / f'{run.name}_sample_workload.csv'
    csv_dest.unlink(missing_ok=True)
    try:
        csv_exported = export_workload_csv(run, csv_dest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        csv_dest.unlink(missing_ok=True)
        csv_exported = None
        print(f'샘플·워크로드 CSV 생성 보류: {exc}', file=sys.stderr)
    latest_csv = output_dir / 'latest_sample_workload.csv'
    if csv_exported:
        shutil.copy2(csv_dest, latest_csv)
    else:
        latest_csv.unlink(missing_ok=True)
    audit_page=run/'report/render_report.html'
    valid,_=integrity(run)
    if audit_page.is_file() and valid:
        rendered=audit_page.read_text(encoding='utf-8')
        dataset_path=run/'report/tta_dataset_independence.json'
        if dataset_path.is_file():
            dataset=load(dataset_path)
            if dataset.get('schema',0)>=2 and '<footer id="footer">' in rendered:
                merged='slo_calibration' not in dataset.get('phase_sample_ids',{})
                comparisons=([('Prime/SLO 보정 ↔ official','prime_vs_official')] if merged else
                             [('Prime ↔ SLO 보정','prime_vs_slo_calibration'),
                              ('Prime ↔ official','prime_vs_official'),
                              ('SLO 보정 ↔ official','slo_calibration_vs_official')])
                counts=dataset.get('pairwise_overlap_counts',{})
                rows=''.join('<tr><td>'+h(label)+'</td><td>'+h(counts.get(key,'Pending'))
                             +'개</td></tr>' for label,key in comparisons)
                panel=('<section><h2>학습 샘플 분리</h2><p>각 기기 partition의 '
                       +('Prime/SLO 보정 행 2–3, official 행 4–5' if merged else
                         'Prime 행 0–1, SLO 보정 행 2–3, official 행 4–5')
                       +'를 사용했습니다. 단계 안에서는 '
                       '매 라운드 같은 행을 다시 읽습니다.</p><div class="scroll"><table>'
                       '<thead><tr><th>비교</th><th>실제 읽힌 토큰 해시 교집합</th></tr></thead><tbody>'
                       +rows+'</tbody></table></div><p class="hint">감사 상태: '
                       +h(dataset.get('status','Pending'))
                       +' · 세부 샘플: report/tta_dataset_samples.csv</p></section>')
                rendered=rendered.replace('<footer id="footer">',panel+'<footer id="footer">',1)
        if csv_exported:
            link = ('<p class="hint"><a href="' + h(csv_dest.name) + '">'
                    '샘플·워크로드 통합 CSV 다운로드</a> · 각 행은 기기 1대의 1라운드입니다.</p>')
            rendered = rendered.replace('<footer id="footer">', link + '<footer id="footer">', 1)
        dest.write_text(rendered,encoding='utf-8')
    else:
        rendered = page(run)
        if csv_exported:
            link = ('<p><a href="' + h(csv_dest.name) + '">'
                    '샘플·워크로드 통합 CSV 다운로드</a></p>')
            rendered = rendered.replace('</main>', link + '</main>', 1)
        dest.write_text(rendered, encoding='utf-8')
    shutil.copy2(dest, output_dir / 'latest.html')
    explanation = output_dir / f'{run.name}_criteria.txt'
    explanation.write_text(criteria_text(run), encoding='utf-8')
    shutil.copy2(explanation, output_dir / 'latest_criteria.txt')
    return dest


def reveal(path):
    if sys.platform == 'darwin':
        try:
            subprocess.run(['open', str(path)], check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10)
        except (OSError, subprocess.SubprocessError):
            print('자동 열기에 실패했습니다. 아래 파일을 브라우저로 여세요.', file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description='LoRACast 현장시험 실행 및 결과 화면')
    sub = parser.add_subparsers(dest='command', required=True)
    run_cmd = sub.add_parser('run', help='3라운드 현장시험 실행 후 결과 표시')
    run_cmd.add_argument('--reference', required=True, type=Path, help='상태 재사용 방식의 SLO 보정 디렉터리')
    run_cmd.add_argument('--mode', choices=HARNESSES, default='reuse',
                         help='reuse: 준비 상태 재사용 / standard: 매번 2라운드 준비')
    run_cmd.add_argument('--config', type=Path)
    run_cmd.add_argument('--allow-legacy', action='store_true',
                         help='이전 산정식의 standard 2+3 시험을 명시적으로 실행 (TTA v2 판정 제외)')
    run_cmd.add_argument('--no-open', action='store_true')
    show_cmd = sub.add_parser('show', help='기존 실행 결과 다시 표시')
    show_cmd.add_argument('--run-dir', required=True, type=Path)
    show_cmd.add_argument('--no-open', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'show':
        dest = publish(args.run_dir)
        print(f'결과 화면: {dest}')
        csv_path = dest.with_name(dest.stem + '_sample_workload.csv')
        if csv_path.is_file():
            print(f'샘플·워크로드 CSV: {csv_path}')
        print(f'기준 설명: {dest.with_name(dest.stem + "_criteria.txt")}')
        if not args.no_open: reveal(dest)
        return 0
    if args.mode == 'standard' and not args.allow_legacy:
        print('standard 2+3 harness는 이전 산정식이며 TTA Onsite v2 증빙이 없습니다. '
              'v2에는 기본 reuse 모드를 사용하세요. 과거 시험만 재현하려면 --allow-legacy를 명시하세요.',
              file=sys.stderr)
        return 2
    harness, default_config = HARNESSES[args.mode]
    config = args.config or harness / default_config
    command = [sys.executable, str(harness / 'orchestrator.py'), '-c', str(config.resolve()),
               'onsite', '--reference', str(args.reference.resolve())]
    run_dir = None
    try:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                if line.startswith('Run: '):
                    run_dir = Path(line.removeprefix('Run: ').strip())
            code = process.wait()
    except KeyboardInterrupt:
        print('\n중단 요청을 전달합니다. 실행 기록을 확인하세요.', file=sys.stderr)
        raise
    if run_dir is None:
        print('실행 디렉터리를 확인하지 못했습니다. 위 오류 메시지를 확인하세요.', file=sys.stderr)
        return code or 2
    dest = publish(run_dir)
    print(f'\n결과 화면: {dest}')
    csv_path = dest.with_name(dest.stem + '_sample_workload.csv')
    if csv_path.is_file():
        print(f'샘플·워크로드 CSV: {csv_path}')
    print(f'기준 설명: {dest.with_name(dest.stem + "_criteria.txt")}')
    print(f'언제든 다시 보기: {sys.executable} {Path(__file__).resolve()} show --run-dir {run_dir}')
    if not args.no_open: reveal(dest)
    return code if (run_dir / 'report/tta_summary.json').is_file() else code or 2


if __name__ == '__main__':
    raise SystemExit(main())
