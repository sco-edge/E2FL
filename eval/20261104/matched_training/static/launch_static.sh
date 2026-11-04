#!/usr/bin/env bash
# Sequential R1 smoke or R2–R100 resume on the isolated RTX host.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo 'usage: launch_static.sh baseline|pyramid smoke|resume' >&2
  exit 2
fi
case "$1" in
  baseline) run_id=nanonx_s42_dfull_qwen_baseline_repeat4x_r50_100_v1 ;;
  pyramid) run_id=nanonx_s42_dfull_qwen_pyramid_profile_slack_repeat4x_r50_100_v1 ;;
  *) echo 'unknown policy' >&2; exit 2 ;;
esac
case "$2" in
  smoke|resume) mode=$2 ;;
  *) echo 'unknown mode' >&2; exit 2 ;;
esac

cd /mnt/ssd4t/loracast_www_fig8_fig9_20261024_wwjang/edge_protocol/matched_repeat4x_static_20261026
exec 9>static_campaign.lock
flock -n 9 || { echo 'another matched static run holds the lock' >&2; exit 75; }
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPU is occupied; refusing to launch' >&2
  exit 75
fi

python_bin=/home/wwjang/flwr/bin/python
"$python_bin" run_static_training.py --validate-only --strict-inputs
"$python_bin" run_static_training.py --run-id "$run_id" --strict-inputs --check-production-prelaunch
run_dir="rtx_runs/$run_id"
if [[ $mode == smoke && -e "$run_dir/state.json" ]]; then
  echo 'smoke target already exists; preserve and inspect it before any resume' >&2
  exit 75
fi
if [[ $mode == resume ]]; then
  if [[ ! -f "$run_dir/state.json" ]]; then
    echo 'R1 smoke state is missing' >&2
    exit 75
  fi
  committed_round=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["round"])' "$run_dir/state.json")
  if (( committed_round < 1 || committed_round > 100 )); then
    echo 'invalid committed round; inspect state before resuming' >&2
    exit 75
  fi
fi

mkdir -p logs
attempt="${run_id}_${mode}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
log="logs/${attempt}.log"
exit_file="logs/${attempt}.exit"
trap 'run_rc=$?; printf "%s\n" "$run_rc" > "$exit_file"' EXIT
if [[ $mode == smoke ]]; then
  LORACAST_MATCHED_STATIC_RTX_APPROVED=1 "$python_bin" -u run_static_training.py \
    --run-id "$run_id" --strict-inputs --stop-after-round 1 > "$log" 2>&1
else
  LORACAST_MATCHED_STATIC_RTX_APPROVED=1 "$python_bin" -u run_static_training.py \
    --run-id "$run_id" --strict-inputs > "$log" 2>&1
fi
echo "$log"
