#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != smoke && $1 != resume ) ]]; then
  echo 'usage: launch_loracast.sh smoke|resume' >&2
  exit 2
fi
mode=$1
cd /mnt/ssd4t/loracast_www_fig8_fig9_20261024_wwjang/edge_protocol/matched_repeat4x_loracast_20261026
run_id=nanonx_s42_dfull_qwen_loracast_floor256_matched_repeat4x_full100
expected=590543fb5481a75294d3270aa4d62dd78e284d3934dd47de83ffa2e30bfbf7c9
actual=$(sha256sum experiment_manifest.json | awk '{print $1}')
if [[ $actual != "$expected" ]]; then
  echo 'frozen manifest SHA mismatch' >&2
  exit 3
fi
exec 9>campaign.lock
flock -n 9 || { echo 'campaign lock occupied' >&2; exit 75; }
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
  echo 'GPU busy' >&2
  exit 75
fi
export LORACAST_NANO_PROTOCOL=/mnt/ssd4t/loracast_www_fig8_fig9_20261024_wwjang/edge_protocol
export LORACAST_MATCHED_RTX_APPROVED=1
python_bin=/home/wwjang/flwr/bin/python
"$python_bin" run_adaptive_training.py --run-id "$run_id" --validate-only
if [[ $mode == smoke ]]; then
  test ! -e "runs/$run_id/state.json"
  "$python_bin" -u run_adaptive_training.py --run-id "$run_id" --stop-after-round 1
else
  test -f "runs/$run_id/state.json"
  "$python_bin" -u run_adaptive_training.py --run-id "$run_id"
fi
