#!/usr/bin/env python3
# This script loads a base Gemma/Qwen model and a LoRA adapter checkpoint for evaluation.
# Before running, please check the following:
# - base_model_id matches the base model used during federated fine-tuning.
# - adapter_ckpt path points to the correct adapter checkpoint for the desired round.
# - LoRA config (r, alpha, dropout, target modules) matches the training setup; specify `--lora-r` accordingly.
# - Device and torch_dtype are appropriate for your hardware.
# - Evaluation task configuration is set up as needed for your use case.

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import math
import random
from typing import List, Dict, Any, Tuple
from datasets import load_dataset
from collections import Counter

import json
import time
import platform

# Base model must match the adapter checkpoint.
# Default stays Gemma; override via --base-model-id for other backbones (e.g., Qwen).
DEFAULT_BASE_MODEL_ID = "google/gemma-3-270m"

# If the Gemma model is gated on Hugging Face, you need an access token that has
# accepted the model license. The recommended options are:
#   1) Run the HF CLI login once on this machine:
#        pip install -U "huggingface_hub[cli]"
#        hf auth login
#      (Then follow the instructions in the terminal.)
#   2) Or export an environment variable, e.g.:
#        export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
#
# If HF_TOKEN is set, it will be passed to from_pretrained; otherwise, the
# huggingface cache/default login will be used.
HF_TOKEN = os.getenv("HF_TOKEN", None)

parser = argparse.ArgumentParser(description="Load a base model and a LoRA adapter checkpoint for evaluation")
parser.add_argument(
    "--adapter-ckpt",
    type=str,
    required=True,
    help="Path to the LoRA adapter checkpoint (.pt) saved by the E2FL/Flower server",
)
parser.add_argument(
    "--lora-r",
    type=int,
    default=None,
    help="LoRA rank r used during training (must match the adapter checkpoint). Required unless --no-adapter is set.",
)
parser.add_argument(
    "--no-adapter",
    action="store_true",
    help="Evaluate the base model only (skip loading LoRA adapter checkpoint)",
)
parser.add_argument(
    "--base-model-id",
    type=str,
    default=DEFAULT_BASE_MODEL_ID,
    help="HF model id for the backbone (must match the adapter checkpoint), e.g., google/gemma-3-270m or Qwen/Qwen2.5-0.5B-Instruct",
)
parser.add_argument(
    "--prompt-format",
    type=str,
    default="auto",
    choices=["auto", "alpaca", "chat", "plain"],
    help="How to format inputs for evaluation. 'alpaca' uses ### Instruction/Response. 'plain' uses Instruction/Answer without ### headers (recommended for Gemma base; stable with BOS/EOS special tokens). 'chat' uses tokenizer.apply_chat_template. 'auto' selects 'alpaca' for Qwen and 'plain' for Gemma, otherwise 'alpaca'. (Use 'chat' explicitly for Qwen-native chat-template eval.)",
)
parser.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned", help="HF dataset name for evaluation")
parser.add_argument("--split", type=str, default="train", help="Dataset split to evaluate (e.g., train/test/validation)")
parser.add_argument(
    "--holdout-ratio",
    type=float,
    default=0.1,
    help="If the dataset has only 'train', then --split test/validation is created by splitting train with this ratio (e.g., 0.1 => 10% holdout).",
)
parser.add_argument(
    "--holdout-seed",
    type=int,
    default=None,
    help="Seed for creating the holdout split when split=test/validation is synthesized from train. Defaults to --seed if not provided.",
)
parser.add_argument(
    "--eval-indices",
    type=str,
    default=None,
    help="(Optional) Path to a JSON file containing a list of example indices to evaluate. If set, overrides random sampling.",
)
parser.add_argument(
    "--save-eval-indices",
    type=str,
    default=None,
    help="(Optional) If set and --eval-indices is not provided, save the sampled indices to this JSON file for reuse.",
)
parser.add_argument("--max-eval-samples", type=int, default=512, help="Max number of examples to evaluate")
parser.add_argument("--eval-batch-size", type=int, default=1, help="Eval batch size")
parser.add_argument("--max-seq-len", type=int, default=128, help="Max sequence length for tokenization")
parser.add_argument(
    "--loss-mode",
    type=str,
    default="response_only",
    choices=["response_only", "full"],
    help="Loss to compute. 'response_only' masks prompt tokens (default). 'full' computes LM loss over prompt+response, masking only padding.",
)
parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling evaluation subset")
parser.add_argument("--do-generate", action="store_true", help="Run a small generation sanity check after loading")
parser.add_argument("--gen-max-new-tokens", type=int, default=64, help="Max new tokens for sanity generation")

# Metadata + saving
parser.add_argument("--round", type=int, default=None, help="(Optional) FL round number associated with this adapter checkpoint")
parser.add_argument("--run-id", type=str, default=None, help="(Optional) Experiment/run identifier (e.g., baseline, loracast)")
parser.add_argument("--save-jsonl", type=str, default=None, help="(Optional) Path to append a JSONL record with evaluation results")
parser.add_argument("--overwrite-jsonl", action="store_true", help="If set, overwrite --save-jsonl instead of appending")

parser.add_argument(
    "--strict-adapter-load",
    action="store_true",
    help="If set, fail when adapter checkpoint keys do not match the PEFT model (missing/unexpected keys).",
)
parser.add_argument(
    "--print-adapter-keys",
    action="store_true",
    help="If set, print a brief summary of adapter checkpoint key patterns for debugging.",
)

args = parser.parse_args()

start_wall = time.time()
start_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_wall))

HOLDOUT_SEED = args.seed if args.holdout_seed is None else args.holdout_seed

adapter_ckpt = args.adapter_ckpt
lora_r = args.lora_r

base_model_id = args.base_model_id
print(f"[evaluate_llm] Base model: {base_model_id}")


def choose_prompt_format(model_id: str, mode: str) -> str:
    if mode != "auto":
        return mode
    mid = model_id.lower()
    # For apples-to-apples comparison with Gemma runs, default to the same
    # Alpaca-style prompt formatting for Qwen as well.
    # NOTE: You can still explicitly select --prompt-format chat if you want
    # Qwen's native chat-template evaluation.
    if "qwen" in mid:
        return "alpaca"
    if "gemma" in mid:
        return "plain"
    return "alpaca"


PROMPT_FORMAT = choose_prompt_format(base_model_id, args.prompt_format)
print(f"[evaluate_llm] Prompt format: {PROMPT_FORMAT}")


def alpaca_to_user_content(example: Dict[str, Any]) -> str:
    inst = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    if inp:
        return f"{inst}\n\nInput:\n{inp}"
    return inst


if (not args.no_adapter) and (lora_r is None):
    raise SystemExit("[evaluate_llm] ERROR: --lora-r is required unless --no-adapter is set")


def format_alpaca(example: Dict[str, Any]) -> Tuple[str, str]:
    """Return (prompt_text, response_text) for Alpaca-style example."""
    inst = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    out = (example.get("output") or "").strip()

    if inp:
        prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{inst}\n\n### Response:\n"

    response = out + "\n"
    return prompt, response


def format_plain(example: Dict[str, Any]) -> Tuple[str, str]:
    """Return (prompt_text, response_text) for a minimal instruction format (no ### headers)."""
    inst = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    out = (example.get("output") or "").strip()

    if inp:
        prompt = f"Instruction:\n{inst}\n\nInput:\n{inp}\n\nAnswer:\n"
    else:
        prompt = f"Instruction:\n{inst}\n\nAnswer:\n"

    response = out + "\n"
    return prompt, response


def format_chat(tokenizer, example: Dict[str, Any]) -> Tuple[str, str]:
    """Return (prompt_text, response_text) but formatted via chat template."""
    user_content = alpaca_to_user_content(example)
    assistant_content = (example.get("output") or "").strip() + "\n"

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_text, assistant_content


def build_batch_with_response_only_labels(
    tokenizer,
    prompts: List[str],
    responses: List[str],
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    """Tokenize prompt+response, masking prompt tokens with -100 so loss is on response only."""
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
            return_tensors=None,
            add_special_tokens=False,
        )
        prompt_lens.append(len(pe["input_ids"]))

    labels = input_ids.clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_batch_with_full_labels(
    tokenizer,
    prompts: List[str],
    responses: List[str],
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    """Tokenize prompt+response, masking only padding tokens with -100 so loss is on full sequence."""
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
    labels[attention_mask == 0] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_split_with_fallback(dataset_name: str, split: str, holdout_seed: int, holdout_ratio: float):
    """Load a HF dataset split. If split is missing but only 'train' exists, create a deterministic holdout."""
    try:
        return load_dataset(dataset_name, split=split)
    except ValueError as e:
        # Common case: datasets like yahma/alpaca-cleaned expose only 'train'
        if split in {"test", "validation", "valid"}:
            base = load_dataset(dataset_name, split="train")
            tt = base.train_test_split(test_size=holdout_ratio, seed=holdout_seed, shuffle=True)
            if split == "test":
                print(f"[evaluate_llm] NOTE: dataset has no '{split}' split; using holdout from train (ratio={holdout_ratio}, seed={holdout_seed})")
                return tt["test"]
            else:
                print(f"[evaluate_llm] NOTE: dataset has no '{split}' split; using holdout from train (ratio={holdout_ratio}, seed={holdout_seed})")
                return tt["test"]
        raise


def evaluate_loss(
    model,
    tokenizer,
    dataset_name: str,
    split: str,
    max_eval_samples: int,
    eval_batch_size: int,
    max_seq_len: int,
    seed: int,
) -> Tuple[float, float, int]:
    """Evaluate average loss and perplexity on an Alpaca-style dataset (supports synthetic holdout). Mode is controlled by args.loss_mode."""
    ds = load_split_with_fallback(dataset_name, split=split, holdout_seed=HOLDOUT_SEED, holdout_ratio=args.holdout_ratio)

    n = len(ds)
    k = min(max_eval_samples, n)

    # Deterministic subset selection
    if args.eval_indices is not None:
        with open(args.eval_indices, "r", encoding="utf-8") as f:
            indices = json.load(f)
        if not isinstance(indices, list) or len(indices) == 0:
            raise ValueError("--eval-indices must be a JSON list of integers")
        indices = [int(i) for i in indices]
        # Filter invalid indices
        indices = [i for i in indices if 0 <= i < n]
        if len(indices) == 0:
            raise ValueError("--eval-indices contained no valid indices for this split")
        indices = indices[:k]
        k = len(indices)
    else:
        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)
        indices = indices[:k]
        if args.save_eval_indices is not None:
            out_p = args.save_eval_indices
            os.makedirs(os.path.dirname(out_p) or ".", exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(indices, f)
            print(f"[evaluate_llm] Saved eval indices -> {out_p} (n={len(indices)})")

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for start in range(0, k, eval_batch_size):
        batch_idx = indices[start : start + eval_batch_size]
        batch = ds.select(batch_idx)

        prompts, responses = [], []
        for ex in batch:
            if PROMPT_FORMAT == "chat":
                p, r = format_chat(tokenizer, ex)
            elif PROMPT_FORMAT == "plain":
                p, r = format_plain(ex)
            else:
                p, r = format_alpaca(ex)
            prompts.append(p)
            responses.append(r)

        if args.loss_mode == "full":
            batch_tensors = build_batch_with_full_labels(tokenizer, prompts, responses, max_seq_len)
        else:
            batch_tensors = build_batch_with_response_only_labels(tokenizer, prompts, responses, max_seq_len)
        batch_tensors = {k: v.to(model.device) for k, v in batch_tensors.items()}

        labels = batch_tensors["labels"]
        loss_tokens = int((labels != -100).sum().item())

        if loss_tokens == 0:
            print(
                f"[evaluate_llm] WARNING: skipping batch with no scored tokens "
                f"at batch start={start}, indices={batch_idx}"
            )
            continue

        with torch.no_grad():
            out = model(**batch_tensors)
            loss = out.loss

        if not torch.isfinite(loss):
            print(
                f"[evaluate_llm] ERROR: non-finite loss "
                f"at batch start={start}, indices={batch_idx}, "
                f"loss_tokens={loss_tokens}, loss={loss.item()}"
            )
            raise RuntimeError("Non-finite evaluation loss detected")

        total_loss += float(loss.item()) * loss_tokens
        total_tokens += loss_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return avg_loss, ppl, total_tokens


device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    token=HF_TOKEN,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    token=HF_TOKEN,
    dtype=torch.float32,
    device_map="auto",
)

if args.no_adapter:
    print("[evaluate_llm] --no-adapter set: evaluating base model without LoRA adapter")
else:
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    state_dict = torch.load(adapter_ckpt, map_location="cpu")
    if args.print_adapter_keys:
        keys = list(state_dict.keys())
        print(f"[evaluate_llm] Adapter state_dict keys: {len(keys)}")
        # show a few examples
        for k in keys[:10]:
            print(f"  key: {k}")
        # pattern summary
        pats = []
        for k in keys:
            if "lora_A" in k:
                pats.append("lora_A")
            elif "lora_B" in k:
                pats.append("lora_B")
            elif "lora" in k:
                pats.append("lora")
            else:
                pats.append("other")
        c = Counter(pats)
        print(f"[evaluate_llm] Key pattern counts: {dict(c)}")
    missing, unexpected = set_peft_model_state_dict(model, state_dict)
    missing = list(missing) if missing is not None else []
    unexpected = list(unexpected) if unexpected is not None else []

    # Only treat missing/unexpected LoRA adapter keys as a mismatch.
    def _is_adapter_key(k: str) -> bool:
        k = str(k)
        return (".lora_A." in k) or (".lora_B." in k) or (".lora_" in k)

    missing_adapter = [k for k in missing if _is_adapter_key(k)]
    unexpected_adapter = [k for k in unexpected if _is_adapter_key(k)]
    missing_base = [k for k in missing if not _is_adapter_key(k)]

    print(
        f"[evaluate_llm] Adapter load summary: missing_adapter={len(missing_adapter)} "
        f"unexpected_adapter={len(unexpected_adapter)} (missing_base={len(missing_base)})"
    )

    if len(missing_adapter) > 0:
        print("[evaluate_llm] Example missing ADAPTER keys (first 10):")
        for k in missing_adapter[:10]:
            print(f"  missing_adapter: {k}")
    else:
        print(f"[evaluate_llm] NOTE: base-model missing keys={len(missing_base)} (expected for adapter-only checkpoints).")

    if len(unexpected_adapter) > 0:
        print("[evaluate_llm] Example unexpected ADAPTER keys (first 10):")
        for k in unexpected_adapter[:10]:
            print(f"  unexpected_adapter: {k}")

    if args.strict_adapter_load and (len(missing_adapter) > 0 or len(unexpected_adapter) > 0):
        raise SystemExit("[evaluate_llm] ERROR: adapter checkpoint did not match PEFT model (see missing/unexpected above).")

    print(f"[evaluate_llm] Loaded LoRA adapter from: {adapter_ckpt}")

#model.to(device)
model.eval()

if args.do_generate:
    if PROMPT_FORMAT == "chat":
        messages = [{"role": "user", "content": "Explain what federated learning is in 2 sentences."}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        if PROMPT_FORMAT == "plain":
            prompt = "Instruction:\nExplain what federated learning is in 2 sentences.\n\nAnswer:\n"
        else:
            prompt = "### Instruction:\nExplain what federated learning is in 2 sentences.\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.gen_max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
    print("\n[evaluate_llm] Generation sanity check:\n")
    # Print only the newly generated tokens (avoid echoing the prompt).
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_ids = out[0][prompt_len:]
    print(tokenizer.decode(gen_ids, skip_special_tokens=True))

print(f"\n[evaluate_llm] Evaluating loss_mode={args.loss_mode} on dataset={args.dataset} split={args.split} ...")
avg_loss, ppl, n_tokens = evaluate_loss(
    model=model,
    tokenizer=tokenizer,
    dataset_name=args.dataset,
    split=args.split,
    max_eval_samples=args.max_eval_samples,
    eval_batch_size=args.eval_batch_size,
    max_seq_len=args.max_seq_len,
    seed=args.seed,
)
metric_name = "avg_full_loss" if args.loss_mode == "full" else "avg_response_loss"
print(f"[evaluate_llm] Done. {metric_name}={avg_loss:.6f}, ppl={ppl:.4f}, loss_tokens={n_tokens}")

end_wall = time.time()
record = {
    "ts": start_iso,
    "wall_time_s": end_wall - start_wall,
    "host": platform.node(),
    "platform": platform.platform(),
    "run_id": args.run_id,
    "round": args.round,
    "base_model_id": base_model_id,
    "adapter_ckpt": adapter_ckpt,
    "no_adapter": bool(args.no_adapter),
    "lora_r": lora_r if not args.no_adapter else None,
    "prompt_format": PROMPT_FORMAT,
    "dataset": args.dataset,
    "split": args.split,
    "holdout_ratio": float(args.holdout_ratio),
    "holdout_seed": int(HOLDOUT_SEED),
    "max_eval_samples": args.max_eval_samples,
    "eval_batch_size": args.eval_batch_size,
    "max_seq_len": args.max_seq_len,
    "loss_mode": args.loss_mode,
    "seed": args.seed,
    "avg_response_loss": float(avg_loss) if args.loss_mode != "full" else None,
    "avg_full_loss": float(avg_loss) if args.loss_mode == "full" else None,
    "ppl": float(ppl),
    "loss_tokens": int(n_tokens),
    "response_tokens": int(n_tokens) if args.loss_mode != "full" else None,
}

if args.save_jsonl:
    out_p = args.save_jsonl
    mode = "w" if args.overwrite_jsonl else "a"
    with open(out_p, mode, encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[evaluate_llm] Saved eval record -> {out_p} (mode={mode})")