#!/usr/bin/env python3
"""Audit immutable validation shards and reconstruct fixed-view losses."""

import argparse
import json
import math
from pathlib import Path

from e2fl.validation_views import selected_views


CONTRACT = ("round", "adapter_ckpt", "adapter_sha256", "partition_manifest",
            "dataset_sha256", "base_model_id", "lora_r", "seed",
            "max_eval_samples", "eval_batch_size", "max_seq_len", "loss_mode",
            "num_shards")


def read_single(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 1:
        raise ValueError(f"expected one shard record: {path}")
    return rows[0]


def combine(manifest_path: Path, prefix: Path, num_shards: int) -> tuple[dict, list[dict]]:
    if num_shards < 2:
        raise ValueError("sharded combination requires at least two shards")
    manifest = json.loads(manifest_path.read_text())
    records = []
    scores = {}
    contract = None
    for shard in range(num_shards):
        stem = f"{prefix}_shard{shard:02d}of{num_shards:02d}"
        record = read_single(Path(stem + ".jsonl"))
        examples = [json.loads(line) for line in Path(stem + ".examples.jsonl").read_text().splitlines()
                    if line]
        if record.get("shard_index") != shard or record.get("num_shards") != num_shards:
            raise ValueError(f"incorrect shard identity: {stem}")
        if record.get("evaluation_method") != "union_shard_token_weighted":
            raise ValueError(f"incorrect evaluation method: {stem}")
        current = {key: record[key] for key in CONTRACT}
        if contract is None:
            contract = current
        elif current != contract:
            raise ValueError(f"validation contract differs: {stem}")
        if record["dataset_sha256"] != manifest["dataset_sha256"]:
            raise ValueError("shard dataset differs from selected manifest")
        if record["partition_manifest"] != str(manifest_path):
            raise ValueError("shard partition manifest path differs")
        if len(examples) != record["unique_examples_scored"]:
            raise ValueError(f"example count mismatch: {stem}")
        views = selected_views(manifest, record["max_eval_samples"], record["seed"])
        all_indices = sorted({index for selected in views.values() for index in selected})
        intended = set(all_indices[shard::num_shards])
        observed = {item["index"] for item in examples}
        if observed != intended or len(observed) != len(examples):
            raise ValueError(f"shard indices incomplete or duplicated: {stem}")
        for item in examples:
            if (not math.isfinite(float(item["loss_sum"]))
                    or int(item["tokens"]) < 1 or item["index"] in scores):
                raise ValueError(f"invalid example loss: {stem}")
            scores[item["index"]] = (float(item["loss_sum"]), int(item["tokens"]))
        records.append(record)
    if set(scores) != set(all_indices):
        raise ValueError("combined shards do not cover the holdout union")
    view_losses = {}
    view_tokens = {}
    for name, selected in views.items():
        numerator = sum(scores[index][0] for index in selected)
        denominator = sum(scores[index][1] for index in selected)
        if denominator < 1:
            raise ValueError(f"view has no scored tokens: {name}")
        view_losses[name] = numerator / denominator
        view_tokens[name] = denominator
    result = {key: records[0][key] for key in CONTRACT if key != "num_shards"}
    result.update({
        "evaluation_method": "union_sharded_token_weighted",
        "shard_count": num_shards,
        "unique_examples_scored": len(scores),
        "view_example_counts": {name: len(selected) for name, selected in views.items()},
        "view_token_counts": view_tokens,
        "global_validation_loss": view_losses["global"],
        "domain_validation_loss": {name.split(":", 1)[1]: value
                                   for name, value in view_losses.items()
                                   if name.startswith("domain:")},
        "client_validation_loss": {name.split(":", 1)[1]: value
                                   for name, value in view_losses.items()
                                   if name.startswith("client:")},
        "worst_domain_validation_loss": max(value for name, value in view_losses.items()
                                            if name.startswith("domain:")),
        "evaluation_wall_s": sum(float(item["evaluation_wall_s"]) for item in records),
    })
    return result, [{"index": index, "loss_sum": scores[index][0],
                     "tokens": scores[index][1]} for index in all_indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--input-prefix", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--save-jsonl", type=Path, required=True)
    parser.add_argument("--save-examples", type=Path, required=True)
    args = parser.parse_args()
    if args.save_jsonl.exists() or args.save_examples.exists():
        parser.error("refusing to overwrite combined validation output")
    result, examples = combine(args.partition_manifest, args.input_prefix, args.num_shards)
    args.save_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.save_jsonl.write_text(json.dumps(result) + "\n")
    args.save_examples.write_text("".join(json.dumps(item) + "\n" for item in examples))
    print(json.dumps({"round": result["round"],
                      "global_validation_loss": result["global_validation_loss"],
                      "unique_examples_scored": len(examples)}))


if __name__ == "__main__":
    main()
