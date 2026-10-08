#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# powermon_daemon_jetson.py 사용 설명
#
# 왜 만들었는가?
# - client_app_llm.py 안의 파워 모니터는 train/upload/eval과 같이
#   "클라이언트 프로세스가 살아 있는 동안"만 전력 샘플을 수집한다.
# - Jetson처럼 빠른 디바이스는 로컬 학습이 끝난 뒤, 느린 디바이스(RPi5)를
#   기다리는 대기 구간(idle/comm 대기)이 길게 존재하는데,
#   이 구간은 기존 per-phase 파워 로그에 잘 드러나지 않는다.
# - 이 스크립트는 FL 실험 전체를 커버하는 "백그라운드 파워 트레이스"를
#   별도 CSV로 남기기 위해 만든 데몬이다.
#
# 어떻게 쓰는가?
# 1) Jetson 쪽에서 E2FL 가상환경을 활성화한다.
#      $ source ~/e2fl/bin/activate        (환경 이름은 실제 설정에 맞게)
#
# 2) FL 실험을 시작하기 전에 이 스크립트를 실행한다.
#      $ python3 powermon_daemon_jetson.py
#
#    - get_device_name()를 통해 device_name (예: jetson-orin, RPi5 등)을
#      자동으로 판단하고, host 이름을 포함해 파워 모니터를 초기화한다.
#    - 측정 결과는 HOME 기준:
#        ~/EEFL/E2FL/eval/power_daemon_{device_name}_{시각}.csv
#      형식의 파일로 저장된다.
#    - 현재 샘플링 주기(freq=0.005) 기준으로 약 5ms 간격으로 샘플링한다.
#
# 3) FL 실험이 모두 끝난 뒤, 터미널에서 Ctrl+C 를 눌러 데몬을 종료한다.
#    - KeyboardInterrupt 발생 시 pm.stop() / pm.save() / pm.close()가 호출되며
#      전체 Duration, 샘플 개수(n_samples), CSV 파일이 출력된다.
#
# 분석 시 주의점
# - 이 데몬 로그는 train/upload/eval 등 phase 경계와는 독립적으로,
#   "디바이스 관점에서의 연속 전력 시계열"을 제공한다.
#   → round-level 로그(fedavg_*.csv)와 timestamp를 기준으로 정렬/정합해서
#     compute/comm/idle 구간을 나눠 쓰는 용도로 사용하면 된다.
# - Jetson만이 아니라 RPi5에서도 동일 패턴으로 데몬을 띄울 수 있으며,
#   powermon_daemon_* 파일만 기기별로 구분해서 사용하면 된다.
#
#(e2fl) netlab@netlab-desktop:~/EEFL/E2FL$ python powermon_daemon_jetson.py 
# -----------------------------------------------------------------------------
from datetime import datetime
import time
import socket
import os
import signal
import sys

from power.powermon import get_power_monitor
from e2fl.utils import get_device_name  # client_app_llm.py에서 쓰는 거랑 동일

home_dir = os.path.expanduser("~")
LOCAL_PATH = f"{home_dir}/EEFL/E2FL/eval/"   # client_app_llm.py와 동일 경로

# Optional: autosave interval (seconds). Set to 0 to disable.
AUTOSAVE_INTERVAL_S = 60

# Optional: stop automatically after this many seconds (0 = run until stopped)
MAX_DURATION_S = 0

# Optional: stop if this sentinel file exists (empty file is fine)
# e.g., touch ~/EEFL/E2FL/eval/STOP_POWER_DAEMON
SENTINEL_STOP_FILE = os.path.join(LOCAL_PATH, "STOP_POWER_DAEMON")

def main():
    # 디바이스 정보 얻기 (jetson_orin_nx, RPi5 구분 등)
    interfaces, device_name, power_info = get_device_name()
    host = socket.gethostname()

    print(f"[DAEMON] device_name={device_name}, host={host}")
    print(f"[DAEMON] power_info={power_info}")

    pm = get_power_monitor(power_info, device_name=host)

    # Ensure output directory exists
    os.makedirs(LOCAL_PATH, exist_ok=True)

    stopped = {"done": False}

    def safe_stop_and_save(reason: str):
        if stopped["done"]:
            return
        stopped["done"] = True
        try:
            print(f"[DAEMON] Stopping power monitor... (reason={reason})")
            elapsed, n_samples = pm.stop()
            print(f"[DAEMON] Duration={elapsed:.3f}s, samples={n_samples}")
        except Exception as e:
            print(f"[DAEMON][WARN] pm.stop() failed: {e}")
        try:
            pm.save(out_fname)
            print(f"[DAEMON] Saved: {out_fname}")
        except Exception as e:
            print(f"[DAEMON][WARN] pm.save() failed: {e}")
        try:
            pm.close()
        except Exception:
            pass
        print("[DAEMON] Exit.")

    def _signal_handler(signum, frame):
        name = signal.Signals(signum).name if signum in [s.value for s in signal.Signals] else str(signum)
        safe_stop_and_save(reason=f"signal:{name}")
        sys.exit(0)

    # Save even if SSH/terminal disconnects (SIGHUP) or container/service stops (SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    try:
        signal.signal(signal.SIGHUP, _signal_handler)
    except Exception:
        pass

    # 파일 이름: 전체 런 커버하게 한 번만 저장
    start_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_fname = os.path.join(
        LOCAL_PATH,
        f"power_daemon_{device_name}_{start_ts}.csv"
    )
    print(f"[DAEMON] Saving to: {out_fname}")
    wall_start = datetime.now()
    print(f"[DAEMON] Wall-clock start: {wall_start.isoformat(timespec='seconds')}")

    # 샘플링 시작 (5ms는 꽤 촘촘, 필요하면 0.01 등으로 늘려도 됨)
    pm.start(freq=0.005)

    last_autosave = time.time()
    start_monotonic = time.time()

    try:
        while True:
            time.sleep(1.0)

            # Optional: stop conditions
            if MAX_DURATION_S and (time.time() - start_monotonic) >= MAX_DURATION_S:
                safe_stop_and_save(reason=f"max_duration:{MAX_DURATION_S}s")
                return

            if SENTINEL_STOP_FILE and os.path.exists(SENTINEL_STOP_FILE):
                safe_stop_and_save(reason=f"sentinel:{SENTINEL_STOP_FILE}")
                return

            # Optional: periodic autosave (best-effort)
            if AUTOSAVE_INTERVAL_S and AUTOSAVE_INTERVAL_S > 0:
                now = time.time()
                if (now - last_autosave) >= AUTOSAVE_INTERVAL_S:
                    try:
                        pm.save(out_fname)
                        print(f"[DAEMON] Autosaved: {out_fname} ({int(now - start_monotonic)}s elapsed)")
                    except Exception as e:
                        print(f"[DAEMON][WARN] Autosave failed: {e}")
                    last_autosave = now

    except KeyboardInterrupt:
        safe_stop_and_save(reason="keyboard_interrupt")
        return

if __name__ == "__main__":
    main()
