#!/usr/bin/env python3
r"""Evaluate per-round LoRA adapter checkpoints in a folder and save PPL (JSONL).

This script is for building learning curves (PPL vs round) by evaluating multiple
adapter checkpoints exported by the FL server.

Your checkpoint filenames look like:
  peft_hf_google_gemma-3-270m_1.pt
  peft_hf_google_gemma-3-270m_2.pt
  ...
(i.e., only the trailing number changes.)

Typical usage (run inside EEFL/E2FL/e2fl):
  python evaluate_llm_round.py \
    --ckpt-dir ../eval/evaluate/20260302/checkpoints \
    --base-model-id google/gemma-3-270m --lora-r 16 \
    --max-seq-len 128 --max-eval-samples 2048 --eval-batch-size 64 --seed 1 \
    --run-id loracast_gemma_full \
    --save-jsonl ../eval/evaluate/20260302/ppl/loracast_gemma_full_rounds.jsonl --overwrite-jsonl

Notes:
  - Loads the base model once and swaps LoRA adapter state dict per checkpoint.
  - Records are written as JSON Lines (JSONL), one line per evaluated checkpoint.
  - Round number is inferred from the trailing number by default (pattern: _(\d+)\.pt$).
  - Supports optional fixed evaluation indices via --eval-indices (load) / --save-eval-indices (save).
  - Supports --loss-mode {response_only,full} to control loss computation (response tokens only or full prompt+response).
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import random
import argparse
import platform
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
try:
    from e2fl.runtime.partition_manifest import dataset_digest, load_manifest
except ModuleNotFoundError as exc:
    if exc.name != "e2fl":
        raise
    from runtime.partition_manifest import dataset_digest, load_manifest


DEFAULT_BASE_MODEL_ID = "google/gemma-3-270m"
HF_TOKEN = os.getenv("HF_TOKEN", None)


def choose_prompt_format(model_id: str, mode: str) -> str:
    if mode != "auto":
        return mode
    mid = model_id.lower()
    if "qwen" in mid:
        return "chat"
    return "alpaca"


def alpaca_to_user_content(example: Dict[str, Any]) -> str:
    inst = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    if inp:
        return f"{inst}\n\nInput:\n{inp}"
    return inst


def format_alpaca(example: Dict[str, Any]) -> Tuple[str, str]:
    inst = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    out = (example.get("output") or "").strip()

    if inp:
        prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{inst}\n\n### Response:\n"

    response = out + "\n"
    return prompt, response


def format_chat(tokenizer, example: Dict[str, Any]) -> Tuple[str, str]:
    user_content = alpaca_to_user_content(example)
    assistant_content = (example.get("output") or "").strip() + "\n"

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_text, assistant_content


def format_loracast(example: Dict[str, Any]) -> str:
    """Match the text constructed by runtime.llm_data._batch_text exactly."""
    pieces = ["Instruction:", str(example.get("instruction", "")).strip()]
    context = str(example.get("input") or "").strip()
    if context:
        pieces.extend(["Context:", context])
    pieces.extend(["Answer:", str(example.get("output", "")).strip()])
    return "\n".join(pieces)


def build_loracast_validation_batch(tokenizer, examples, max_seq_len: int):
    encoded = tokenizer(
        [format_loracast(example) for example in examples],
        return_tensors="pt", padding=True, truncation=True,
        max_length=max_seq_len,
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    encoded["labels"] = labels
    return encoded


def build_batch_with_response_only_labels(
    tokenizer,
    prompts: List[str],
    responses: List[str],
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    full_texts = [p + r for p, r in zip(prompts, responses)]
    enc = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=False,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    prompt_lens = []
    for p in prompts:
        pe = tokenizer(
            p,
            padding=False,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
            return_tensors=None,
        )
        prompt_lens.append(len(pe["input_ids"]))

    labels = input_ids.clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# --- New: Helper for full loss (no masking, labels = input_ids) ---
def build_batch_with_full_labels(
    tokenizer,
    prompts: List[str],
    responses: List[str],
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    full_texts = [p + r for p, r in zip(prompts, responses)]
    enc = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# --- New: Helpers for loading/saving eval indices ---
def load_eval_indices(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "indices" in obj:
        return obj["indices"]
    elif isinstance(obj, list):
        return obj
    else:
        raise ValueError(f"Invalid eval indices format in {path}")

def save_eval_indices(path: str, indices: List[int]) -> None:
    obj = {"indices": indices, "n": len(indices)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def evaluate_loss(
    model,
    tokenizer,
    dataset_name: str,
    split: str,
    max_eval_samples: int,
    eval_batch_size: int,
    max_seq_len: int,
    seed: int,
    prompt_format: str,
    holdout_ratio: Optional[float] = None,
    holdout_seed: int = 123,
    *,
    eval_indices: Optional[List[int]] = None,
    loss_mode: str = "response_only",
    evaluation_dataset=None,
) -> Tuple[float, float, int, List[int]]:
    if evaluation_dataset is not None:
        ds = evaluation_dataset
    else:
        try:
            ds = load_dataset(dataset_name, split=split)
        except ValueError as e:
            if "Unknown split" in str(e) and split == "test" and holdout_ratio is not None:
                full_ds = load_dataset(dataset_name, split="train")
                split_ds = full_ds.train_test_split(test_size=holdout_ratio, seed=holdout_seed, shuffle=True)
                ds = split_ds["test"]
                print(f"[evaluate_llm_round] NOTE: dataset has no '{split}' split; using holdout from train (ratio={holdout_ratio}, seed={holdout_seed})")
            else:
                raise

    n = len(ds)
    if eval_indices is not None:
        k = min(max_eval_samples, len(eval_indices), n)
        indices = eval_indices[:k]
    else:
        k = min(max_eval_samples, n)
        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)
        indices = indices[:k]

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for start in range(0, k, eval_batch_size):
        batch_idx = indices[start : start + eval_batch_size]
        batch = ds.select(batch_idx)

        if prompt_format == "loracast":
            batch_tensors = build_loracast_validation_batch(tokenizer, batch, max_seq_len)
            batch_tensors = {key: value.to(model.device) for key, value in batch_tensors.items()}
            with torch.no_grad():
                out = model(**batch_tensors)
                loss = out.loss
            labels = batch_tensors["labels"]
            loss_tokens = int((labels[:, 1:] != -100).sum().item())
        else:
            prompts, responses = [], []
            for ex in batch:
                if prompt_format == "chat":
                    p, r = format_chat(tokenizer, ex)
                else:
                    p, r = format_alpaca(ex)
                prompts.append(p)
                responses.append(r)

            if loss_mode == "full":
                batch_tensors = build_batch_with_full_labels(tokenizer, prompts, responses, max_seq_len)
                batch_tensors = {key: value.to(model.device) for key, value in batch_tensors.items()}
                with torch.no_grad():
                    out = model(**batch_tensors)
                    loss = out.loss
                loss_tokens = int((batch_tensors["attention_mask"] == 1).sum().item())
            else:
                batch_tensors = build_batch_with_response_only_labels(tokenizer, prompts, responses, max_seq_len)
                batch_tensors = {key: value.to(model.device) for key, value in batch_tensors.items()}
                with torch.no_grad():
                    out = model(**batch_tensors)
                    loss = out.loss
                labels = batch_tensors["labels"]
                loss_tokens = int((labels != -100).sum().item())

        total_loss += float(loss.item()) * loss_tokens
        total_tokens += loss_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return avg_loss, ppl, total_tokens, indices


def infer_round_from_name(name: str, pattern: str | None) -> int | None:
    if pattern is None:
        pattern = r"_(\d+)\.pt$"  # default: trailing number
    m = re.search(pattern, name, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None



def main():
    ap = argparse.ArgumentParser(description="Evaluate all adapter checkpoints in a folder and save JSONL")
    ap.add_argument("--ckpt-dir", type=str, required=True, help="Directory containing adapter .pt checkpoints")
    ap.add_argument("--base-model-id", type=str, default=DEFAULT_BASE_MODEL_ID)
    ap.add_argument("--prompt-format", type=str, default="auto", choices=["auto", "alpaca", "chat", "loracast"])
    ap.add_argument("--lora-r", type=int, default=None, help="LoRA rank (required)")

    ap.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--holdout-ratio", type=float, default=None, help="If split is missing (e.g., test), use a holdout from train")
    ap.add_argument("--holdout-seed", type=int, default=123, help="Seed for holdout split")
    ap.add_argument("--max-eval-samples", type=int, default=512)
    ap.add_argument("--eval-batch-size", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--save-jsonl", type=str, required=True)
    ap.add_argument("--overwrite-jsonl", action="store_true")
    ap.add_argument("--round-regex", type=str, default=None, help="Regex with one capturing group for round")

    # New arguments for eval indices and loss mode
    ap.add_argument("--loss-mode", type=str, default="response_only", choices=["response_only", "full"], help="Compute loss on response tokens only (default) or on full prompt+response")
    ap.add_argument("--eval-indices", type=str, default=None, help="Path to JSON file with fixed evaluation indices")
    ap.add_argument("--save-eval-indices", type=str, default=None, help="If set, save the sampled eval indices to this JSON file (only used when --eval-indices is not provided)")
    ap.add_argument("--partition-manifest", type=str, default=None, help="Evaluate the disjoint global and domain holdouts from a partition manifest")

    args = ap.parse_args()

    if args.lora_r is None:
        raise SystemExit("[evaluate_llm_round] ERROR: --lora-r is required")
    if args.partition_manifest and (args.eval_indices or args.holdout_ratio is not None or args.split != "train"):
        raise SystemExit("--partition-manifest requires --split train without --eval-indices or --holdout-ratio")
    if args.partition_manifest:
        # The training loader uses full causal-LM labels and its own instruction
        # format. Keep the experimental validation objective identical.
        args.prompt_format = "loracast"
        args.loss_mode = "full"

    manifest = load_manifest(args.partition_manifest) if args.partition_manifest else None
    manifest_dataset = None
    domain_indices = {}
    client_indices = {}
    if manifest:
        if manifest["dataset_name"] != args.dataset:
            raise ValueError("partition manifest dataset does not match --dataset")
        manifest_dataset = load_dataset(
            args.dataset, split="train", revision=manifest.get("dataset_revision")
        )
        if len(manifest_dataset) != manifest["dataset_size"] or dataset_digest(manifest_dataset) != manifest["dataset_sha256"]:
            raise ValueError("evaluation dataset content differs from partition manifest")
        for offset, (domain, indices) in enumerate(sorted(manifest["validation"]["domain_indices"].items())):
            if indices:
                domain_indices[domain] = list(indices)
                random.Random(args.seed + offset + 1).shuffle(domain_indices[domain])
        for client_id, entry in sorted(manifest["validation"].get("clients", {}).items()):
            if entry["indices"]:
                client_indices[client_id] = list(entry["indices"])
                random.Random(args.seed + int(client_id) + 1001).shuffle(client_indices[client_id])

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(ckpt_dir)

    prompt_format = choose_prompt_format(args.base_model_id, args.prompt_format)
    print(f"[evaluate_llm_round] Base model: {args.base_model_id}")
    print(f"[evaluate_llm_round] Prompt format: {prompt_format}")
    print(f"[evaluate_llm_round] Checkpoint dir: {ckpt_dir}")

    out_p = Path(args.save_jsonl)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    mode0 = "w" if args.overwrite_jsonl else "a"

    # Load tokenizer/model once
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        token=HF_TOKEN,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # Build PEFT model once
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.eval()

    ckpts = sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        raise ValueError(f"No .pt checkpoints found in: {ckpt_dir}")

    items = []
    for p in ckpts:
        rnd = infer_round_from_name(p.name, args.round_regex)
        if manifest and rnd is None:
            continue  # Experiment quality curves contain numbered global checkpoints only.
        items.append((rnd, p))

    # sort by round (unknown rounds at the end)
    items.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 10**9, t[1].name))

    # --- Handle fixed eval indices (load/save) ---
    if manifest:
        fixed_indices = list(manifest["validation"]["indices"])
        random.Random(args.seed).shuffle(fixed_indices)
    elif args.eval_indices:
        fixed_indices = load_eval_indices(args.eval_indices)
    else:
        fixed_indices = None
    indices_saved = False
    saved_indices_path = args.save_eval_indices if args.save_eval_indices and not args.eval_indices else None
    indices_used_for_save = None

    wrote_any = False
    for idx, (rnd, pth) in enumerate(items):
        start_wall = time.time()
        start_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_wall))

        state_dict = torch.load(pth, map_location="cpu")
        set_peft_model_state_dict(model, state_dict)

        avg_loss, ppl, n_tokens, indices_used = evaluate_loss(
            model=model,
            tokenizer=tokenizer,
            dataset_name=args.dataset,
            split=args.split,
            max_eval_samples=args.max_eval_samples,
            eval_batch_size=args.eval_batch_size,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
            prompt_format=prompt_format,
            holdout_ratio=args.holdout_ratio,
            holdout_seed=args.holdout_seed,
            eval_indices=fixed_indices,
            loss_mode=args.loss_mode,
            evaluation_dataset=manifest_dataset,
        )
        domain_losses = {}
        for domain, indices in domain_indices.items():
            domain_loss, _, _, _ = evaluate_loss(
                model=model, tokenizer=tokenizer, dataset_name=args.dataset,
                split=args.split, max_eval_samples=args.max_eval_samples,
                eval_batch_size=args.eval_batch_size, max_seq_len=args.max_seq_len,
                seed=args.seed, prompt_format=prompt_format,
                eval_indices=indices, loss_mode=args.loss_mode,
                evaluation_dataset=manifest_dataset,
            )
            domain_losses[domain] = float(domain_loss)
        client_losses = {}
        for client_id, indices in client_indices.items():
            client_loss, _, _, _ = evaluate_loss(
                model=model, tokenizer=tokenizer, dataset_name=args.dataset,
                split=args.split, max_eval_samples=args.max_eval_samples,
                eval_batch_size=args.eval_batch_size, max_seq_len=args.max_seq_len,
                seed=args.seed, prompt_format=prompt_format,
                eval_indices=indices, loss_mode=args.loss_mode,
                evaluation_dataset=manifest_dataset,
            )
            client_losses[client_id] = float(client_loss)
        # Save indices if requested and not yet saved (only on first round)
        if saved_indices_path and not indices_saved:
            save_eval_indices(saved_indices_path, indices_used)
            indices_saved = True
        # For record
        indices_used_for_save = indices_used

        rec = {
            "ts": start_iso,
            "wall_time_s": time.time() - start_wall,
            "host": platform.node(),
            "platform": platform.platform(),
            "run_id": args.run_id,
            "round": rnd,
            "base_model_id": args.base_model_id,
            "adapter_ckpt": str(pth),
            "no_adapter": False,
            "lora_r": int(args.lora_r),
            "prompt_format": prompt_format,
            "dataset": args.dataset,
            "split": args.split,
            "max_eval_samples": args.max_eval_samples,
            "eval_batch_size": args.eval_batch_size,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
            "loss_mode": args.loss_mode,
            "eval_indices": str(args.eval_indices) if args.eval_indices else None,
            "save_eval_indices": str(args.save_eval_indices) if args.save_eval_indices else None,
            "ppl": float(ppl),
            "loss_tokens": int(n_tokens),
            "holdout_ratio": args.holdout_ratio,
            "holdout_seed": int(args.holdout_seed),
        }
        if manifest:
            rec["partition_manifest"] = str(args.partition_manifest)
            rec["dataset_sha256"] = manifest["dataset_sha256"]
            rec["global_validation_loss"] = float(avg_loss)
            rec["domain_validation_loss"] = domain_losses
            rec["client_validation_loss"] = client_losses
            rec["worst_domain_validation_loss"] = max(domain_losses.values()) if domain_losses else None
        if args.loss_mode == "full":
            rec["avg_full_loss"] = float(avg_loss)
        else:
            rec["avg_response_loss"] = float(avg_loss)
            rec["response_tokens"] = int(n_tokens)  # backward compat

        mode = mode0 if (idx == 0 and not wrote_any) else "a"
        with open(out_p, mode, encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        wrote_any = True

        rn_disp = "?" if rnd is None else str(rnd)
        print(f"[evaluate_llm_round] round={rn_disp} mode={args.loss_mode} ppl={ppl:.4f} loss={avg_loss:.6f} tokens={n_tokens} ckpt={pth.name}")

    print(f"[evaluate_llm_round] Saved {len(items)} records -> {out_p}")


if __name__ == "__main__":
    main()
