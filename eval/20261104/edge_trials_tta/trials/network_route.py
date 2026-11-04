#!/usr/bin/env python3
"""Manage only the Jetson's temporary /32 Wi-Fi route to the TTA server."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time

LEASE=Path(__file__).with_name('jetson_wifi_route_lease.json')
# Keep the control channel independent of the Wi-Fi path being repaired.
# These are the same Jetson SSH options used by the onsite harness.
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
     '-o','ConnectTimeout=8','-J','pi5_5',
     '-o','HostName=192.168.128.89',
     '-o','HostKeyAlias=192.168.128.88','jetson']
DEST='192.168.128.124'
INTERFACE='wlP1p1s0'
SOURCE='192.168.128.88'


def remote(command):
    result=subprocess.run(SSH+[command],capture_output=True,text=True,timeout=20)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f'SSH exit {result.returncode}')
    return result.stdout.strip()


def route():
    return json.loads(remote(f'ip -j route get {DEST}'))[0]


def exact_route():
    return json.loads(remote(f'ip -j route show {DEST}/32'))


def clear_lost_lease():
    if not LEASE.is_file():
        return False
    lease=json.loads(LEASE.read_text())
    if not lease.get('active') or exact_route():
        return False
    # NetworkManager can remove an iproute2 route when the Wi-Fi driver resets.
    # A missing exact route cannot be ours to delete or preserve.
    lease.update(active=False,lost_at=time.time(),lost_reason='exact route disappeared')
    LEASE.write_text(json.dumps(lease,indent=2))
    print('이전 임시 Wi-Fi 경로가 사라져 기록을 정리했습니다.')
    return True


def prepare():
    clear_lost_lease()
    current=route()
    if current.get('dev')==INTERFACE and current.get('prefsrc')==SOURCE:
        print('Jetson Wi-Fi route ready:',current)
        return
    if LEASE.exists() and json.loads(LEASE.read_text()).get('active'):
        raise RuntimeError('기존 임시 경로 기록이 있으나 실제 경로가 다릅니다. 수동 확인이 필요합니다.')
    if exact_route():
        raise RuntimeError('서버 전용 /32 경로가 이미 있어 덮어쓰지 않습니다.')
    if remote(f'test -d /sys/class/net/{INTERFACE}/wireless && echo yes')!='yes':
        raise RuntimeError('Jetson Wi-Fi 인터페이스를 확인할 수 없습니다.')
    addresses=json.loads(remote(f'ip -j address show dev {INTERFACE}'))
    if not addresses or SOURCE not in [x.get('local') for x in addresses[0].get('addr_info',[])]:
        raise RuntimeError('Jetson Wi-Fi 주소가 설정과 다릅니다.')
    # A forced-interface TCP probe checks AP forwarding before changing route.
    probe=("python3 -c 'import socket; s=socket.socket(); "
           f"s.setsockopt(socket.SOL_SOCKET,socket.SO_BINDTODEVICE,b\"{INTERFACE}\\0\"); "
           f"s.settimeout(4); s.connect((\"{DEST}\",22)); s.close()'")
    remote(probe)
    remote(f'sudo -n ip route add {DEST}/32 dev {INTERFACE} src {SOURCE}')
    after=route()
    if after.get('dev')!=INTERFACE or after.get('prefsrc')!=SOURCE:
        raise RuntimeError('임시 경로 추가 후 Wi-Fi 경로 확인에 실패했습니다.')
    LEASE.write_text(json.dumps({'active':True,'created_at':time.time(),
                                 'destination':DEST,'interface':INTERFACE,
                                 'source':SOURCE,'previous_route':current},indent=2))
    print('Jetson Wi-Fi route ready:',after)


def restore():
    if not LEASE.is_file():
        print('이 도구가 생성한 임시 경로 기록이 없습니다.');return
    lease=json.loads(LEASE.read_text())
    if not lease.get('active'):
        print('이미 복구된 임시 경로입니다.');return
    if clear_lost_lease():
        print('제거할 임시 경로가 이미 없습니다.');return
    current=route()
    if current.get('dev')!=lease['interface'] or current.get('prefsrc')!=lease['source']:
        raise RuntimeError('현재 경로가 생성 당시와 달라 자동 제거를 거부합니다.')
    remote(f'sudo -n ip route del {DEST}/32 dev {INTERFACE}')
    lease.update(active=False,restored_at=time.time())
    LEASE.write_text(json.dumps(lease,indent=2))
    print('임시 Wi-Fi 경로 제거 완료. 현재 경로:',route())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['status','prepare','restore'])
    args=parser.parse_args()
    try:
        if args.command=='prepare':prepare()
        elif args.command=='restore':restore()
        else:print(json.dumps(route(),indent=2))
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as e:
        parser.exit(2,f'ERROR: {e}\n')


if __name__=='__main__': main()
