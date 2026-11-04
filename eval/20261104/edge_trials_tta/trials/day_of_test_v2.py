#!/usr/bin/env python3
"""Day-of-test workflow outside the fingerprinted LoRACast harness.

One successful SLO calibration is selected per Seoul calendar day. The
calibration artifact stays untouched; this file only records its identity and
passes it to the existing onsite run_adaptive_training.
"""
from __future__ import annotations

import argparse
import builtins
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import uuid
from zoneinfo import ZoneInfo

from tta_trial_v2 import ROOT, integrity
from network_route import prepare as prepare_wifi_route, restore as restore_wifi_route

HARNESS = ROOT / 'edge_trials/harness_v2'
DEFAULT_CONFIG = HARNESS / 'suite.toml'
SESSIONS = Path(__file__).resolve().parent / 'day_sessions_v2'
GENERATED_CONFIGS = Path(__file__).resolve().parent / 'generated_configs_v2'


def display_path(value):
    """Use ~ as the display prefix for the project; keep stored paths intact."""
    text = str(value)
    prefixes = [str(ROOT)]
    if ROOT.is_relative_to(Path.home()):
        prefixes.append('~/' + str(ROOT.relative_to(Path.home())))
    for prefix in prefixes:
        text = re.sub(re.escape(prefix) + r"(?=/|$)", "~", text)
    return re.sub(re.escape(str(Path.home())) + r"(?=/|$)", "~", text)


def display_print(*values, **kwargs):
    builtins.print(*(display_path(value) for value in values), **kwargs)


def materialize_config(base, model=None, dataset=None, directory=GENERATED_CONFIGS):
    """Freeze optional model/dataset overrides without changing the base suite."""
    base = Path(base).resolve()
    if model is None and dataset is None:
        return base
    original = base.read_text()
    parsed = tomllib.loads(original)
    profile = parsed.get('profile_common', {})
    overrides = {'model': model, 'dataset': dataset}
    for key, value in overrides.items():
        if value is not None and (not value.strip() or '\n' in value or '\r' in value):
            raise ValueError(f'{key}은 비어 있지 않은 한 줄 문자열이어야 합니다.')
    start = re.search(r'(?m)^\[profile_common\][ \t]*$', original)
    if start is None:
        raise ValueError('설정 파일에 [profile_common]이 없습니다.')
    end = re.search(r'(?m)^\[', original[start.end():])
    stop = start.end() + end.start() if end else len(original)
    section = original[start.end():stop]
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in profile:
            raise ValueError(f'[profile_common]에 {key} 항목이 없습니다.')
        pattern = rf'(?m)^{re.escape(key)}[ \t]*=[^\n]*$'
        section, count = re.subn(pattern, key + ' = ' + json.dumps(value, ensure_ascii=False), section)
        if count != 1:
            raise ValueError(f'[profile_common]의 {key} 항목을 하나만 찾을 수 있어야 합니다.')
    rendered = original[:start.end()] + section + original[stop:]
    expected = {key: value if value is not None else profile.get(key)
                for key, value in overrides.items()}
    actual = tomllib.loads(rendered)['profile_common']
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError('모델·데이터셋 설정 생성 결과가 요청값과 다릅니다.')
    digest = hashlib.sha256(rendered.encode()).hexdigest()
    path = Path(directory) / f'{base.stem}_{digest[:16]}.toml'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != rendered:
            raise ValueError('생성된 설정 파일이 이미 변경되었습니다.')
    else:
        with path.open('x') as stream:
            stream.write(rendered)
    return path.resolve()


def seoul_now():
    return datetime.now(ZoneInfo('Asia/Seoul'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_fingerprint(config):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0,str(HARNESS))
    from orchestrator import fingerprint
    return fingerprint(tomllib.loads(Path(config).read_text()))


def selected_path(date=None, config=DEFAULT_CONFIG):
    """Keep separate same-day SLO choices for distinct immutable protocols."""
    return SESSIONS / f'{date or seoul_now().date()}_{current_fingerprint(config)[:12]}.json'


def existing_selected_path(date=None, config=DEFAULT_CONFIG):
    current=selected_path(date,config)
    if current.exists():
        return current
    legacy=SESSIONS / f'{date or seoul_now().date()}.json'
    if legacy.is_file():
        record=json.loads(legacy.read_text())
        if record.get('test_fingerprint') == current_fingerprint(config):
            return legacy
    return current


def stream(command):
    """Forward progress and retain the last attempt in case of a harness retry."""
    run = None
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            display_print(line, end='', flush=True)
            if line.startswith('Run: '):
                run = Path(line.removeprefix('Run: ').strip()).resolve()
        code = process.wait()
    return code, run


def check_calibration(run):
    run = Path(run).resolve()
    if not run.is_dir():
        raise ValueError(f'보정 실행 디렉터리가 없습니다: {run}')
    okay, reason = integrity(run)
    if not okay:
        raise ValueError(f'보정 증거 확인 실패: {reason}')
    state = json.loads((run / 'state.json').read_text())
    if state.get('status') != 'validated' or state.get('stage') not in ('calibration','prime'):
        raise ValueError('완료된 SLO 보정 실행이 아닙니다.')
    lock = json.loads((run / 'slo_lock.json').read_text())
    merged=lock.get('slo_source')=='prime'
    if (lock.get('warmup_rounds') != 0 or
            lock.get('measurement_rounds') != (2 if merged else 5) or
            state['stage'] != ('prime' if merged else 'calibration')):
        raise ValueError('SLO 보정 라운드 수 또는 실행 단계가 잠금 기록과 다릅니다.')
    if not isinstance(lock.get('slo_joules'), dict) or set(lock['slo_joules']) != {'RPi5','jetson_orin_nx'}:
        raise ValueError('Pi·Jetson 에너지 상한이 모두 필요합니다.')
    if not lock.get('test_fingerprint') or not lock.get('state_sha256'):
        raise ValueError('보정 출처 정보가 불완전합니다.')
    return lock


def load_selected(date=None, config=DEFAULT_CONFIG):
    path = existing_selected_path(date,config)
    if not path.is_file():
        raise ValueError(f'오늘 고정한 보정이 없습니다: {path}. 먼저 prepare를 실행하세요.')
    selected = json.loads(path.read_text())
    if selected.get('schema') != 1 or selected.get('date') != str(date or seoul_now().date()):
        raise ValueError('당일 보정 선택 기록의 날짜가 맞지 않습니다.')
    run = Path(selected['calibration_run']).resolve()
    lock = check_calibration(run)
    if sha(run / 'slo_lock.json') != selected['slo_lock_sha256']:
        raise ValueError('선택 후 SLO 잠금 파일이 바뀌었습니다.')
    if sha(run / 'checksums.json') != selected['checksums_sha256']:
        raise ValueError('선택 후 보정 증거 목록이 바뀌었습니다.')
    if lock['test_fingerprint'] != selected['test_fingerprint']:
        raise ValueError('선택 후 시험 설정 지문이 바뀌었습니다.')
    if lock['state_sha256'] != selected['state_sha256']:
        raise ValueError('선택 후 예측 상태 지문이 바뀌었습니다.')
    config = Path(selected['config_path']).resolve()
    if not config.is_file() or sha(config) != selected['config_sha256']:
        raise ValueError('보정 당시 시험 설정 파일이 바뀌었습니다.')
    if current_fingerprint(config) != selected['test_fingerprint']:
        raise ValueError('보정 이후 TTA 계측 코드 지문이 바뀌었습니다. 기존 기준으로 시험할 수 없습니다.')
    return selected, lock


def select_calibration(run, config, prime_run=None, replace=False):
    """Register an already validated same-day calibration without rerunning it."""
    run=Path(run).resolve(); config=Path(config).resolve()
    lock=check_calibration(run)
    now=seoul_now()
    if datetime.fromisoformat(lock['created_at']).astimezone(ZoneInfo('Asia/Seoul')).date() != now.date():
        raise ValueError('SLO 잠금 시각이 오늘이 아닙니다.')
    if lock['test_fingerprint'] != current_fingerprint(config):
        raise ValueError('보정 실행과 현재 설정·코드 지문이 다릅니다.')
    current=existing_selected_path(now.date(),config)
    previous_bytes = current.read_bytes() if current.exists() else None
    if current.exists() and not replace:
        selected, existing_lock=load_selected(now.date(),config)
        if Path(selected['calibration_run']).resolve() != run:
            raise ValueError('같은 조건의 당일 SLO가 이미 선택되어 있습니다.')
        print_status(selected,existing_lock)
        return 0
    selected=selected_path(now.date(),config)
    record={
        'schema':1,'date':str(now.date()),'selected_at':now.isoformat(),
        'calibration_run':str(run),'prime_run':str(prime_run.resolve()) if prime_run else None,
        'config_path':str(config),'config_sha256':sha(config),
        'slo_lock_sha256':sha(run/'slo_lock.json'),
        'checksums_sha256':sha(run/'checksums.json'),
        'test_fingerprint':lock['test_fingerprint'],'state_sha256':lock['state_sha256'],
    }
    SESSIONS.mkdir(parents=True,exist_ok=True)
    if replace and previous_bytes is not None:
        archive=SESSIONS/'history'/f'{current.stem}_{uuid.uuid4().hex}.json'
        archive.parent.mkdir(parents=True,exist_ok=True)
        with archive.open('xb') as stream:
            stream.write(previous_bytes)
        record['supersedes_selection']=str(archive.resolve())
        selected=current
    payload=json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)
    if replace and previous_bytes is not None:
        if current.read_bytes()!=previous_bytes:
            raise ValueError('보정 선택이 동시에 변경되어 교체를 중단했습니다.')
        temporary=selected.with_name(selected.name+'.'+uuid.uuid4().hex+'.tmp')
        temporary.write_text(payload)
        temporary.replace(selected)
        display_print('이전 보정 선택 기록 보존:',archive)
    else:
        with selected.open('x') as stream:
            stream.write(payload)
    print_status(record,lock)
    display_print('감시관 방문 전에 위 SLO와 해시를 보관하세요. 현장에서는 onsite만 실행합니다.')
    return 0


def print_status(selected, lock):
    run = Path(selected['calibration_run']).resolve()
    display_print('\n당일 보정 고정 완료')
    display_print('날짜 (한국):', selected['date'])
    display_print('보정 실행:', run)
    display_print('SLO 보정 파일:', run / 'slo_lock.json')
    display_print('보정 학습 CSV:', run / 'fedavg.csv')
    if lock.get('slo_source')=='prime':
        display_print('Prime 실제 학습 샘플:', run / 'prime_samples.jsonl')
        display_print('단계 간 샘플 감사: official 완료 후 report/tta_dataset_independence.json')
    else:
        display_print('Prime·보정 샘플 감사:', run / 'report' / 'tta_dataset_independence.json')
        display_print('실제 학습 샘플 CSV:', run / 'report' / 'tta_dataset_samples.csv')
    display_print('Pi 전력 CSV (분석용):', run / 'rpi5' / 'power_aligned.csv')
    display_print('Jetson 전력 CSV (분석용):', run / 'jetson' / 'power_aligned.csv')
    display_print('Pi 전력 CSV (원본):', run / 'rpi5' / 'power.csv')
    display_print('Jetson 전력 CSV (원본):', run / 'jetson' / 'power.csv')
    display_print('증거 체크섬:', run / 'checksums.sha256')
    display_print('보정 시각:', lock.get('created_at', '기록 없음'))
    display_print('Pi 5 상한:', f"{lock['slo_joules']['RPi5']:.2f} J")
    display_print('Jetson 상한:', f"{lock['slo_joules']['jetson_orin_nx']:.2f} J")
    display_print('SLO SHA-256:', selected['slo_lock_sha256'])
    display_print('SLO 산정:', '보정 평균 에너지 ×', lock['margin'])
    if lock.get('slo_source')=='prime':
        display_print('prime/SLO 보정: 1~2라운드')
    else:
        display_print('prime 보정: 1~2라운드 · SLO 보정:', lock['measurement_rounds'], '라운드')


def prepare(args):
    selected = existing_selected_path(config=args.config)
    force=getattr(args,'force',False)
    if force and args.state:
        raise ValueError('--force는 새 보정을 실행하므로 --state와 함께 사용할 수 없습니다.')
    if selected.exists() and not force:
        record, lock = load_selected(config=args.config)
        print_status(record, lock)
        return 0
    prepare_wifi_route()
    prime_run=args.state.resolve() if args.state else None
    if prime_run is None:
        display_print('새 TTA v2 코드 지문에 맞는 2라운드 prime을 실행합니다.',flush=True)
        code,prime_run=stream([sys.executable,str(HARNESS/'orchestrator.py'),
                               '-c',str(args.config.resolve()),'prime'])
        if code or prime_run is None:
            display_print(f'prime 실패. 실행 기록: {prime_run or "생성되지 않음"}',file=sys.stderr)
            return code or 2
    cfg=tomllib.loads(Path(args.config).read_text())
    if cfg['suite'].get('slo_source')=='prime':
        display_print('prime/SLO 보정: 1~2라운드',flush=True)
        return select_calibration(prime_run,args.config,prime_run,replace=force)
    command = [sys.executable, str(HARNESS / 'orchestrator.py'),
               '-c', str(args.config.resolve()), 'calibrate-slo',
               '--state', str(prime_run)]
    display_print('당일 SLO 보정을 시작합니다. 성공한 기준은 한 번만 선택합니다.', flush=True)
    code, run = stream(command)
    if code or run is None:
        display_print(f'보정 실패. 실행 기록: {run or "생성되지 않음"}', file=sys.stderr)
        return code or 2
    if seoul_now().date() != datetime.fromisoformat(
            json.loads((run/'slo_lock.json').read_text())['created_at']).astimezone(
                ZoneInfo('Asia/Seoul')).date():
        raise ValueError('자정이 지나 날짜가 바뀌었습니다. 당일 보정 절차를 다시 시작하세요.')
    return select_calibration(run,args.config,prime_run,replace=force)


def onsite(args):
    rounds = getattr(args, 'rounds', 3)
    if rounds < 1:
        raise ValueError('본시험 라운드 수는 1 이상이어야 합니다.')
    selected, lock = load_selected(config=args.config)
    prepare_wifi_route()
    print_status(selected, lock)
    command = [sys.executable, str(Path(__file__).with_name('tta_trial_v2.py')),
               'run', '--reference', selected['calibration_run'],
               '--config', selected['config_path'], '--rounds', str(rounds)]
    if args.no_open:
        command.append('--no-open')
    return stream(command)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description='LoRACast 당일 보정 및 현장시험')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='감시관 방문 전: prime 2라운드에서 예측기와 에너지 SLO를 함께 보정')
    p.add_argument('--force', action='store_true', help='새 보정 성공 후 기준 교체; 기존 보정과 시험 증거 보존')
    p.add_argument('--state', type=Path,
                   help='새 코드 지문으로 검증된 2라운드 prime 디렉터리; 생략 시 자동 prime')
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--model', help='시험 모델 이름; 예: hf:google/gemma-3-270m')
    p.add_argument('--dataset', help='시험 데이터셋 이름; 예: yahma/alpaca-cleaned')
    p = sub.add_parser('select', help='이미 검증한 당일 보정 실행을 새 코드 지문에 연결')
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--model')
    p.add_argument('--dataset')
    p = sub.add_parser('status', help='오늘 고정한 보정 기준 확인')
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--model')
    p.add_argument('--dataset')
    sub.add_parser('finish', help='시험 종료 후 이 도구가 추가한 Jetson 임시 Wi-Fi 경로 제거')
    p = sub.add_parser('onsite', help='감시관 입회 시: 고정 기준으로 3라운드 시험')
    p.add_argument('--rounds', type=int, default=3, help='본시험 측정 라운드 수 (기본: 3)')
    p.add_argument('--no-open', action='store_true')
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--model')
    p.add_argument('--dataset')
    args = parser.parse_args(argv)
    try:
        if args.command != 'finish':
            args.config = materialize_config(args.config, args.model, args.dataset)
        if args.command == 'prepare':
            return prepare(args)
        if args.command == 'select':
            return select_calibration(args.reference,args.config)
        if args.command == 'status':
            print_status(*load_selected(config=args.config))
            return 0
        if args.command == 'finish':
            restore_wifi_route()
            return 0
        return onsite(args)
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        display_print('ERROR:', exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
