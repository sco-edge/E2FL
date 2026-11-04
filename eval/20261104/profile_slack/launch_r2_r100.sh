#!/usr/bin/env bash
set -euo pipefail
cd /mnt/ssd4t/loracast_www_fig8_fig9_20261024_wwjang/edge_protocol/profile_slack_component_v1
exec 9>run.lock
flock -n 9 || { echo 'another profile-slack run_adaptive_training holds the lock' >&2; exit 75; }
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPU is occupied; refusing to launch' >&2
  exit 75
fi
test ! -e r2_r100_exit_code.txt
trap 'run_rc=$?; printf "%s\n" "$run_rc" > r2_r100_exit_code.txt' EXIT
/home/wwjang/flwr/bin/python -u run_profile_slack.py > r2_r100.log 2>&1
