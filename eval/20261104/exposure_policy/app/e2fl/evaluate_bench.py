#!/usr/bin/env python3
"""Benchmark saved FL adapters with lm-evaluation-harness.

This script assumes your FL training saved LoRA adapter *state_dict* checkpoints as a single `.pt` file
(e.g., `hf_google_gemma-3-270m_yahma_alpaca-cleaned_adapter.pt`).

`lm-eval-harness` expects PEFT adapters in a directory created by `PeftModel.save_pretrained()`.
So we:
  1) load the base model,
  2) wrap with the matching LoRA config,
  3) load the adapter state_dict,
  4) save a PEFT adapter directory,
  5) run `lm_eval --model hf --model_args pretrained=...,peft=<dir>`.

References:
- lm-eval-harness supports PEFT adapters by adding `,peft=PATH` to `--model_args`. See README.  
  https://github.com/EleutherAI/lm-evaluation-harness

Usage examples:
  # Dry-run prints commands without executing
  python evaluate_bench.py --dry-run

  # Convert+run benchmarks
  python evaluate_bench.py --device cuda:0 --batch-size auto --tasks hellaswag,piqa,arc_easy

  # Only run (skip conversion) if adapter dirs already exist
  python evaluate_bench.py --skip-convert

  # Use a custom cache location (recommended if system disk is small)
  python evaluate_bench.py --hf-home /data/hf_cache --device cuda:0

  # Apply chat template for instruct/chat models (recommended for Qwen Instruct)
  python evaluate_bench.py --apply-chat-template auto
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict


# ------------------------
# Experiment registry
# ------------------------

@dataclass(frozen=True)
class AdapterSpec:
    name: str
    base_model_id: str
    adapter_pt: Path
    lora_r: int
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


DEFAULT_SPECS: List[AdapterSpec] = [
    AdapterSpec(
        name="gemma_e2fl",
        base_model_id="google/gemma-3-270m",
        adapter_pt=Path("../eval/evaluate/20260302/9. E2FL (LLM Perf)/hf_google_gemma-3-270m_yahma_alpaca-cleaned_adapter.pt"),
        lora_r=16,
    ),
    AdapterSpec(
        name="gemma_baseline",
        base_model_id="google/gemma-3-270m",
        adapter_pt=Path("../eval/evaluate/20260302/10. Baseline (LLM Perf)/hf_google_gemma-3-270m_yahma_alpaca-cleaned_adapter.pt"),
        lora_r=16,
    ),
    AdapterSpec(
        name="qwen_baseline",
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_pt=Path("../eval/evaluate/20260302/11. Baseline (Qwen)/hf_Qwen_Qwen2.5-0.5B-Instruct_yahma_alpaca-cleaned_adapter.pt"),
        lora_r=16,
    ),
    AdapterSpec(
        name="qwen_e2fl",
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_pt=Path("../eval/evaluate/20260302/12. E2FL (Qwen)/hf_Qwen_Qwen2.5-0.5B-Instruct_yahma_alpaca-cleaned_adapter.pt"),
        lora_r=16,
    ),
]


# ------------------------
# Helpers
# ------------------------

def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def ensure_cmd_exists(cmd: str) -> None:
    if shutil.which(cmd) is None:
        raise SystemExit(
            f"[evaluate_bench] ERROR: '{cmd}' not found in PATH. "
            f"Install lm-eval-harness (e.g., pip install -U 'lm_eval[hf]') and retry."
        )


def convert_pt_to_peft_dir(
    spec: AdapterSpec,
    out_root: Path,
    dtype: str,
    force: bool,
) -> Path:
    """Convert adapter state_dict (.pt) into a PEFT adapter directory.

    Returns the created directory path.
    """
    out_dir = out_root / f"{_safe_name(spec.name)}__{_safe_name(spec.base_model_id)}__r{spec.lora_r}"

    if out_dir.exists() and not force:
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer (useful to keep alongside adapter dir)
    tok = AutoTokenizer.from_pretrained(spec.base_model_id, cache_dir=os.environ.get("HF_HUB_CACHE"))

    # Load base model on CPU for conversion to avoid consuming VRAM
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.float16)

    base = AutoModelForCausalLM.from_pretrained(
        spec.base_model_id,
        device_map="cpu",
        dtype=torch_dtype,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
    )

    lora_cfg = LoraConfig(
        r=spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        target_modules=list(spec.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base, lora_cfg)

    sd = torch.load(spec.adapter_pt, map_location="cpu")
    set_peft_model_state_dict(model, sd)

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    return out_dir


def run_lm_eval(
    name: str,
    base_model_id: str,
    peft_dir: Optional[Path],
    tasks: List[str],
    device: str,
    batch_size: str,
    num_fewshot: int,
    out_root: Path,
    extra_model_args: Optional[str],
    dry_run: bool,
    apply_chat_template: bool,
    fewshot_as_multiturn: bool,
) -> Tuple[Path, List[str]]:
    """Run lm_eval for one model(+optional PEFT adapter).

    Returns (output_dir, cmd_list).
    """

    out_dir = out_root / _safe_name(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_args_parts = [f"pretrained={base_model_id}"]
    if peft_dir is not None:
        model_args_parts.append(f"peft={str(peft_dir)}")

    # Allow caller to pass things like: dtype=float16,trust_remote_code=True,revision=...
    if extra_model_args:
        # accept both comma-separated or already formatted
        extra = extra_model_args.strip().lstrip(",")
        if extra:
            model_args_parts.append(extra)

    model_args = ",".join(model_args_parts)

    cmd = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--device",
        device,
        "--batch_size",
        batch_size,
        "--output_path",
        str(out_dir),
    ]

    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]

    if apply_chat_template:
        cmd.append("--apply_chat_template")

    if fewshot_as_multiturn:
        cmd.append("--fewshot_as_multiturn")

    print("\n[evaluate_bench] Running:")
    print("  " + " ".join(cmd))

    if not dry_run:
        # Inherit env so HF_* cache settings are applied to datasets downloads.
        subprocess.run(cmd, check=True, env=os.environ.copy())

    return out_dir, cmd


def parse_lm_eval_summary(out_dir: Path) -> Dict[str, float]:
    """Best-effort parse of lm_eval JSON results under out_dir.

    lm_eval writes one or more JSON files. We'll merge task metrics of interest.
    """
    summary: Dict[str, float] = {}
    json_files = list(out_dir.glob("*.json"))
    if not json_files:
        return summary

    # Heuristic: pick the largest json (usually contains results)
    json_files.sort(key=lambda p: p.stat().st_size, reverse=True)

    try:
        data = json.loads(json_files[0].read_text())
    except Exception:
        return summary

    # Common layout: {"results": {task: {metric: value, ...}}, ...}
    results = data.get("results", {}) if isinstance(data, dict) else {}
    if not isinstance(results, dict):
        return summary

    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        # Keep common accuracy-like metrics
        for k in ("acc", "acc_norm", "exact_match", "f1"):
            if k in metrics and isinstance(metrics[k], (int, float)):
                summary[f"{task}.{k}"] = float(metrics[k])

    return summary


# ------------------------
# Main
# ------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--tasks",
        type=str,
        default="hellaswag,piqa,arc_easy",
        help="Comma-separated lm-eval tasks",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device passed to lm_eval (e.g., cuda:0 or cpu)",
    )
    ap.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="lm_eval --batch_size (e.g., auto, auto:4, 8, 16)",
    )
    ap.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="lm_eval --num_fewshot",
    )
    ap.add_argument(
        "--apply-chat-template",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help=(
            "Control lm-eval-harness chat templating. "
            "'auto' enables chat template for likely instruct/chat models (e.g., *Instruct*). "
            "'on' forces it for all models, 'off' disables it."
        ),
    )
    ap.add_argument(
        "--fewshot-as-multiturn",
        action="store_true",
        help="Pass --fewshot_as_multiturn to lm_eval (requires --apply-chat-template and --num-fewshot > 0).",
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="dtype used during adapter conversion (CPU).",
    )
    ap.add_argument(
        "--extra-model-args",
        type=str,
        default="dtype=float16",
        help="Extra lm_eval model_args appended (comma-separated), e.g. dtype=float16,trust_remote_code=True",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default="~/bench_results",
        help="Output root directory for converted adapters + lm_eval results",
    )
    ap.add_argument(
        "--hf-home",
        type=str,
        default=None,
        help="Set HF_HOME (Hugging Face cache root). If unset, uses <out-root>/hf_cache.",
    )
    ap.add_argument(
        "--hf-datasets-cache",
        type=str,
        default=None,
        help="Set HF_DATASETS_CACHE. If unset, uses <out-root>/hf_cache/datasets.",
    )
    ap.add_argument(
        "--hf-hub-cache",
        type=str,
        default=None,
        help="Set HF_HUB_CACHE. If unset, uses <out-root>/hf_cache/hub.",
    )
    ap.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip .pt -> PEFT dir conversion (assumes adapter dirs already exist)",
    )
    ap.add_argument(
        "--force-convert",
        action="store_true",
        help="Re-convert even if PEFT dir already exists",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands but do not execute",
    )

    args = ap.parse_args()

    ensure_cmd_exists("lm_eval")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    out_root = Path(args.out_root).resolve()

    # Configure Hugging Face caches under out_root by default to avoid filling system disk.
    hf_home = Path(args.hf_home).expanduser().resolve() if args.hf_home else (out_root / "hf_cache")
    hf_datasets_cache = Path(args.hf_datasets_cache).expanduser().resolve() if args.hf_datasets_cache else (hf_home / "datasets")
    hf_hub_cache = Path(args.hf_hub_cache).expanduser().resolve() if args.hf_hub_cache else (hf_home / "hub")

    hf_home.mkdir(parents=True, exist_ok=True)
    hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    hf_hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets_cache)
    os.environ["HF_HUB_CACHE"] = str(hf_hub_cache)
    # Transformers also respects TRANSFORMERS_CACHE / HUGGINGFACE_HUB_CACHE in some setups.
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_hub_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_hub_cache))

    print(f"[evaluate_bench] HF_HOME={hf_home}")
    print(f"[evaluate_bench] HF_DATASETS_CACHE={hf_datasets_cache}")
    print(f"[evaluate_bench] HF_HUB_CACHE={hf_hub_cache}")

    peft_root = out_root / "peft_adapters"
    eval_root = out_root / "lm_eval"
    peft_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = out_root / f"summary_{timestamp}.csv"
    summary_json = out_root / f"summary_{timestamp}.json"

    all_rows: List[Dict[str, object]] = []

    # Also benchmark base models (no adapter)
    base_only = [
        ("gemma_base", "google/gemma-3-270m"),
        ("qwen_base", "Qwen/Qwen2.5-0.5B-Instruct"),
    ]

    def _should_apply_chat_template(model_id: str) -> bool:
        if args.apply_chat_template == "on":
            return True
        if args.apply_chat_template == "off":
            return False
        # auto: heuristics for instruct/chat variants
        mid = model_id.lower()
        return ("instruct" in mid) or ("chat" in mid)

    if args.apply_chat_template != "off":
        print(f"[evaluate_bench] apply_chat_template={args.apply_chat_template} (auto applies to *Instruct*/*chat* models)")

    for name, base_model_id in base_only:
        out_dir, cmd = run_lm_eval(
            name=name,
            base_model_id=base_model_id,
            peft_dir=None,
            tasks=tasks,
            device=args.device,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            out_root=eval_root,
            extra_model_args=args.extra_model_args,
            dry_run=args.dry_run,
            apply_chat_template=_should_apply_chat_template(base_model_id),
            fewshot_as_multiturn=args.fewshot_as_multiturn,
        )
        metrics = parse_lm_eval_summary(out_dir) if not args.dry_run else {}
        row = {
            "name": name,
            "base_model": base_model_id,
            "adapter_pt": "",
            "peft_dir": "",
            "tasks": ",".join(tasks),
            "cmd": " ".join(cmd),
            **metrics,
        }
        all_rows.append(row)

    for spec in DEFAULT_SPECS:
        if not spec.adapter_pt.exists():
            print(f"[evaluate_bench] WARN: adapter not found: {spec.adapter_pt}")
            continue

        peft_dir = peft_root / f"{_safe_name(spec.name)}__{_safe_name(spec.base_model_id)}__r{spec.lora_r}"
        if not args.skip_convert:
            peft_dir = convert_pt_to_peft_dir(
                spec=spec,
                out_root=peft_root,
                dtype=args.dtype,
                force=args.force_convert,
            )
            print(f"[evaluate_bench] PEFT dir ready: {peft_dir}")
        else:
            if not peft_dir.exists():
                raise SystemExit(
                    f"[evaluate_bench] ERROR: --skip-convert was set but PEFT dir does not exist: {peft_dir}\n"
                    f"Run without --skip-convert once to generate it."
                )

        out_dir, cmd = run_lm_eval(
            name=spec.name,
            base_model_id=spec.base_model_id,
            peft_dir=peft_dir,
            tasks=tasks,
            device=args.device,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            out_root=eval_root,
            extra_model_args=args.extra_model_args,
            dry_run=args.dry_run,
            apply_chat_template=_should_apply_chat_template(spec.base_model_id),
            fewshot_as_multiturn=args.fewshot_as_multiturn,
        )
        metrics = parse_lm_eval_summary(out_dir) if not args.dry_run else {}
        row = {
            "name": spec.name,
            "base_model": spec.base_model_id,
            "adapter_pt": str(spec.adapter_pt),
            "peft_dir": str(peft_dir),
            "lora_r": spec.lora_r,
            "lora_alpha": spec.lora_alpha,
            "lora_dropout": spec.lora_dropout,
            "target_modules": ",".join(spec.target_modules),
            "tasks": ",".join(tasks),
            "cmd": " ".join(cmd),
            **metrics,
        }
        all_rows.append(row)

    # Write summary
    # Flatten all keys for CSV
    all_keys = sorted({k for row in all_rows for k in row.keys()})

    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    summary_json.write_text(json.dumps(all_rows, indent=2))

    print("\n[evaluate_bench] Done.")
    print(f"  Summary CSV : {summary_csv}")
    print(f"  Summary JSON: {summary_json}")
    print(f"  lm_eval outputs are under: {eval_root}")


if __name__ == "__main__":
    main()