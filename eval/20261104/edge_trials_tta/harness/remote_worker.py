"""Detached, bounded process supervisor; only stops its own process group."""
import json, os, signal, subprocess, sys, time
from pathlib import Path

job = Path(sys.argv[1]).resolve()
cfg = json.loads(job.read_text())
root = job.parent
name = job.name.removesuffix('.job.json')
state = root / (name + '.state.json')

def save(obj):
    tmp = state.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj))
    tmp.replace(state)

stopping = False
def stop(*_):
    global stopping
    stopping = True
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
start = time.time()
if (root / (name + '.stop')).exists():
    save(dict(supervisor_pid=os.getpid(), start=start, end=time.time(), status='stopped_before_start'))
    raise SystemExit(0)
with (root / (name + '.log')).open('ab', buffering=0) as log:
    p = subprocess.Popen(cfg['argv'], cwd=cfg['cwd'], env={**os.environ, **cfg.get('env', {})},
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    info = dict(pid=p.pid, supervisor_pid=os.getpid(), start=start, status='running')
    save(info)
    reason = None
    while p.poll() is None:
        if stopping or (root / (name + '.stop')).exists():
            reason = 'stopped'; break
        if time.time() - start > cfg['timeout_s']:
            reason = 'timeout'; break
        time.sleep(.2)
    # Also reap children left behind by a normally exiting CLI.
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        p.wait(timeout=8)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait()
    info.update(end=time.time(), returncode=p.returncode, status=reason or ('success' if p.returncode == 0 else 'failed'))
    save(info)
