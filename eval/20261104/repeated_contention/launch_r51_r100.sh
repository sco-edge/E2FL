#!/usr/bin/env bash
set -uo pipefail

cd /mnt/ssd4t/loracast_www_fig8_fig9_20261024_wwjang/edge_protocol/repeated_contention_r50_100_20261026 || exit 92
run_id=nanonx_s42_dfull_qwen_loracast_floor256_repeat4x_r50_100_20261026
exit_file=r51_r100.exit
trap 'status=$?; printf "%s\n" "$status" > "$exit_file"' EXIT

printf '%s\n' '9402c9c5bb3bf48f678aaf22f859cbad79dec8224f24dfd17e4fbf87c69f321c  branch_run_adaptive_training.py' \
  'c8a9b67c33ce900f58932d3a48267b3568499a7083c9b92780a8b5c4043b63de  experiment_manifest.json' \
  | sha256sum -c - || exit 90

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPU already has a compute process; refusing branch launch' >&2
  exit 91
fi

LORACAST_REPEAT_CONTENTION_RTX_APPROVED=1 /home/wwjang/flwr/bin/python branch_run_adaptive_training.py \
  --run-id "$run_id"
status=$?
exit "$status"
