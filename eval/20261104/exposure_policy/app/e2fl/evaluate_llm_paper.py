#!/usr/bin/env python3
"""Evaluate one global LoRA checkpoint on fixed manifest views in one pass.

The global, proxy-domain, and client-associated views may overlap. Scoring
their union once avoids repeated GPU work and yields per-example numerators
for auditing the token-weighted causal-LM loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from e2fl.evaluate_llm_round import build_loracast_validation_batch
from e2fl.runtime.partition_manifest import dataset_digest, load_manifest
from e2fl.validation_views import selected_views


def evaluate(checkpoint: Path, manifest_path: Path, cap: int, batch_size: int,
             max_seq_len: int, seed: int, base_model_id: str,
             lora_rank: int, shard_index: int = 0,
             num_shards: int = 1) -> tuple[dict, list[dict]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid validation shard")
    started = time.time()
    manifest = load_manifest(manifest_path)
    dataset = load_dataset(manifest["dataset_name"], split="train",
                           revision=manifest.get("dataset_revision"))
    if len(dataset) != manifest["dataset_size"] or dataset_digest(dataset) != manifest["dataset_sha256"]:
        raise ValueError("evaluation dataset differs from partition manifest")
    views = selected_views(manifest, cap, seed)
    all_indices = sorted({index for selected in views.values() for index in selected})
    indices = all_indices[shard_index::num_shards]
    if not indices:
        raise ValueError("manifest has no validation examples")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, dtype=torch.bfloat16, device_map="auto")
    model = get_peft_model(model, LoraConfig(
        r=lora_rank, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    set_peft_model_state_dict(model, state)
    model.eval()
    scores: dict[int, tuple[float, int]] = {}
    with torch.no_grad():
        for position in range(0, len(indices), batch_size):
            chosen = indices[position:position + batch_size]
            batch = build_loracast_validation_batch(
                tokenizer, dataset.select(chosen), max_seq_len)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            labels = batch["labels"][:, 1:]
            shifted = logits[:, :-1, :].float()
            losses = F.cross_entropy(
                shifted.reshape(-1, shifted.shape[-1]), labels.reshape(-1),
                ignore_index=-100, reduction="none").reshape(labels.shape)
            token_counts = (labels != -100).sum(dim=1)
            loss_sums = losses.sum(dim=1)
            for index, loss_sum, tokens in zip(chosen, loss_sums.tolist(), token_counts.tolist()):
                if tokens <= 0 or not math.isfinite(loss_sum):
                    raise ValueError(f"invalid example loss at index {index}")
                scores[index] = (float(loss_sum), int(tokens))
            print(f"scored {len(scores)}/{len(indices)} holdout examples", flush=True)
    examples = [{"index": index, "loss_sum": scores[index][0],
                 "tokens": scores[index][1]} for index in indices]
    if num_shards > 1:
        return {
            "round": int(checkpoint.stem.rsplit("_", 1)[1]),
            "adapter_ckpt": str(checkpoint),
            "adapter_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "partition_manifest": str(manifest_path),
            "dataset_sha256": manifest["dataset_sha256"],
            "base_model_id": base_model_id,
            "lora_r": lora_rank,
            "seed": seed,
            "max_eval_samples": cap,
            "eval_batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "loss_mode": "full",
            "evaluation_method": "union_shard_token_weighted",
            "shard_index": shard_index,
            "num_shards": num_shards,
            "unique_examples_scored": len(indices),
            "evaluation_wall_s": time.time() - started,
        }, examples
    view_losses = {}
    view_tokens = {}
    for name, selected in views.items():
        numerator = sum(scores[index][0] for index in selected)
        denominator = sum(scores[index][1] for index in selected)
        if denominator <= 0:
            raise ValueError(f"view has no scored tokens: {name}")
        view_losses[name] = numerator / denominator
        view_tokens[name] = denominator
    domain_losses = {name.split(":", 1)[1]: value for name, value in view_losses.items()
                     if name.startswith("domain:")}
    client_losses = {name.split(":", 1)[1]: value for name, value in view_losses.items()
                     if name.startswith("client:")}
    record = {
        "round": int(checkpoint.stem.rsplit("_", 1)[1]),
        "adapter_ckpt": str(checkpoint),
        "partition_manifest": str(manifest_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "base_model_id": base_model_id,
        "seed": seed,
        "max_eval_samples": cap,
        "eval_batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "loss_mode": "full",
        "evaluation_method": "union_single_pass_token_weighted",
        "unique_examples_scored": len(indices),
        "view_example_counts": {name: len(selected) for name, selected in views.items()},
        "view_token_counts": view_tokens,
        "global_validation_loss": view_losses["global"],
        "domain_validation_loss": domain_losses,
        "client_validation_loss": client_losses,
        "worst_domain_validation_loss": max(domain_losses.values()),
        "evaluation_wall_s": time.time() - started,
    }
    return record, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--base-model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-eval-samples", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--save-jsonl", type=Path, required=True)
    parser.add_argument("--save-examples", type=Path)
    args = parser.parse_args()
    if min(args.lora_r, args.max_eval_samples, args.eval_batch_size,
           args.max_seq_len) < 1 or args.seed < 0:
        parser.error("invalid rank, sample, batch, sequence, or seed setting")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid validation shard")
    if args.num_shards > 1 and args.save_examples is None:
        parser.error("sharded evaluation requires --save-examples")
    if args.save_jsonl.exists() or (args.save_examples and args.save_examples.exists()):
        parser.error("refusing to overwrite evaluation output")
    record, examples = evaluate(args.checkpoint, args.partition_manifest,
                                args.max_eval_samples, args.eval_batch_size,
                                args.max_seq_len, args.seed,
                                args.base_model_id, args.lora_r,
                                args.shard_index, args.num_shards)
    args.save_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.save_jsonl.write_text(json.dumps(record) + "\n")
    if args.save_examples:
        args.save_examples.parent.mkdir(parents=True, exist_ok=True)
        args.save_examples.write_text("".join(json.dumps(item) + "\n" for item in examples))
    print(json.dumps({"round": record["round"],
                      "global_validation_loss": record.get("global_validation_loss"),
                      "unique_examples_scored": len(examples)}))


if __name__ == "__main__":
    main()
