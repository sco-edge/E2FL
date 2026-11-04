"""Reproducible, disjoint partitions for LoRACast learning experiments.

The default LoRACast data path does not use this module. Experiments opt in by
passing a manifest path to ``llm_data.load_data``. Alpaca-Cleaned has no domain
column, so ``instruction_rules_v1`` assigns transparent task-type labels from
the instruction text. The manifest includes examples for a manual label audit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Mapping, Sequence


DOMAIN_RULES = (
    ("code", re.compile(r"\b(code|program|python|javascript|java|sql|script|function|algorithm|debug|html|regex)\b", re.I)),
    ("math", re.compile(r"\b(calculate|compute|equation|algebra|arithmetic|probability|percentage|geometry|math|solve for|how many)\b", re.I)),
    ("translation", re.compile(r"\b(translate|translation|in (?:spanish|french|german|chinese|japanese|korean)|from (?:spanish|french|german|chinese|japanese|korean))\b", re.I)),
    ("summarization", re.compile(r"\b(summarize|summarise|summary|briefly recap|key points of)\b", re.I)),
    ("creative", re.compile(r"\b(write|compose|draft|create)\b.*\b(story|poem|song|essay|email|letter|dialogue|speech)\b", re.I)),
)


def domain_of(example: Mapping[str, object]) -> str:
    """Assign a task-type domain using only the instruction, never the answer."""
    instruction = str(example.get("instruction", ""))
    for name, pattern in DOMAIN_RULES:
        if pattern.search(instruction):
            return name
    return "general"


def dataset_digest(examples: Sequence[Mapping[str, object]]) -> str:
    """Hash row order and training fields so all hosts use the same dataset."""
    digest = hashlib.sha256()
    for example in examples:
        row = [str(example.get(key, "")) for key in ("instruction", "input", "output")]
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _counts_from_proportions(size: int, proportions: Sequence[float]) -> list[int]:
    raw = [size * value for value in proportions]
    counts = [math.floor(value) for value in raw]
    remainder = size - sum(counts)
    order = sorted(range(len(counts)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in order[:remainder]:
        counts[i] += 1
    return counts


def _split_validation(domains: Sequence[str], fraction: float, seed: int) -> tuple[list[int], dict[str, list[int]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, domain in enumerate(domains):
        grouped[domain].append(index)
    rng = random.Random(seed)
    by_domain: dict[str, list[int]] = {}
    for domain in sorted(grouped):
        indices = grouped[domain][:]
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        by_domain[domain] = sorted(indices[:count])
    return sorted(index for indices in by_domain.values() for index in indices), by_domain


def build_manifest(
    examples: Sequence[Mapping[str, object]],
    *,
    dataset_name: str,
    num_clients: int,
    mode: str,
    seed: int,
    validation_fraction: float = 0.1,
    dataset_revision: str | None = None,
    min_client_fraction: float = 0.02,
) -> dict:
    """Build IID, mild, or strong Non-IID partitions with one fixed holdout.

    Mild and strong use per-domain Dirichlet allocation with alpha 1.0 and 0.1.
    They change actual task-type proportions, not merely shuffle each client.
    """
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")
    if mode not in {"iid", "mild", "strong"}:
        raise ValueError("mode must be iid, mild, or strong")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie between 0 and 0.5")
    if not 0 < min_client_fraction < 1 / num_clients:
        raise ValueError("min_client_fraction must be positive and below 1/num_clients")

    domains = [domain_of(example) for example in examples]
    validation, validation_by_domain = _split_validation(domains, validation_fraction, seed)
    validation_set = set(validation)
    train_by_domain: dict[str, list[int]] = defaultdict(list)
    for index, domain in enumerate(domains):
        if index not in validation_set:
            train_by_domain[domain].append(index)
    train_total = sum(map(len, train_by_domain.values()))
    if train_total < num_clients:
        raise ValueError("not enough training records for every client")

    minimum = max(1, math.ceil(train_total * min_client_fraction))
    alpha = {"mild": 1.0, "strong": 0.1}.get(mode)
    assignments: list[list[int]] | None = None
    for attempt in range(200):
        rng = random.Random(seed + 1009 * (attempt + 1))
        candidates = [[] for _ in range(num_clients)]
        if mode == "iid":
            indices = [index for group in train_by_domain.values() for index in group]
            rng.shuffle(indices)
            for offset, index in enumerate(indices):
                candidates[offset % num_clients].append(index)
        else:
            for domain in sorted(train_by_domain):
                indices = train_by_domain[domain][:]
                rng.shuffle(indices)
                draws = [rng.gammavariate(alpha, 1.0) for _ in range(num_clients)]
                total = sum(draws)
                proportions = [value / total for value in draws]
                counts = _counts_from_proportions(len(indices), proportions)
                cursor = 0
                for client_id, count in enumerate(counts):
                    candidates[client_id].extend(indices[cursor : cursor + count])
                    cursor += count
        if all(len(indices) >= minimum for indices in candidates):
            assignments = [sorted(indices) for indices in candidates]
            break
    if assignments is None:
        raise ValueError("could not satisfy minimum client dataset size; reduce min_client_fraction")

    clients = {}
    for client_id, indices in enumerate(assignments):
        clients[str(client_id)] = {
            "train_indices": indices,
            "dataset_size": len(indices),
            "domain_counts": dict(sorted(Counter(domains[index] for index in indices).items())),
        }

    # Reserve a client-associated holdout in addition to the global/domain
    # views. Allocate each domain according to the training domain ownership,
    # while keeping the client holdouts disjoint and their union global.
    client_validation: list[list[int]] = [[] for _ in range(num_clients)]
    for domain, indices in sorted(validation_by_domain.items()):
        owned = [clients[str(i)]["domain_counts"].get(domain, 0) for i in range(num_clients)]
        proportions = [count / sum(owned) for count in owned]
        counts = _counts_from_proportions(len(indices), proportions)
        shuffled = indices[:]
        random.Random(seed + 7919 + sum(map(ord, domain))).shuffle(shuffled)
        cursor = 0
        for client_id, count in enumerate(counts):
            client_validation[client_id].extend(shuffled[cursor : cursor + count])
            cursor += count
    validation_clients = {
        str(client_id): {
            "indices": sorted(indices),
            "domain_counts": dict(sorted(Counter(domains[index] for index in indices).items())),
        }
        for client_id, indices in enumerate(client_validation)
    }

    manifest = {
        "schema_version": 2,
        "dataset_name": dataset_name,
        "dataset_revision": dataset_revision,
        "dataset_size": len(examples),
        "dataset_sha256": dataset_digest(examples),
        "domain_method": "instruction_rules_v1",
        "mode": mode,
        "dirichlet_alpha": alpha,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "num_clients": num_clients,
        "clients": clients,
        "validation": {
            "indices": validation,
            "domain_indices": validation_by_domain,
            "domain_counts": {name: len(indices) for name, indices in validation_by_domain.items()},
            "clients": validation_clients,
        },
        "domain_audit_examples": {
            name: [str(examples[index].get("instruction", "")) for index in indices[:5]]
            for name, indices in sorted(train_by_domain.items())
        },
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, object]) -> None:
    """Reject overlap, missing rows, duplicate rows, and malformed client IDs."""
    if manifest.get("schema_version") not in (1, 2):
        raise ValueError("unsupported partition manifest schema")
    size = int(manifest["dataset_size"])
    clients = manifest["clients"]
    expected_clients = int(manifest["num_clients"])
    if set(clients) != {str(i) for i in range(expected_clients)}:
        raise ValueError("client IDs must be consecutive from zero")
    validation = list(manifest["validation"]["indices"])
    train = [index for client in clients.values() for index in client["train_indices"]]
    all_indices = validation + train
    if len(all_indices) != size or set(all_indices) != set(range(size)):
        raise ValueError("partition indices must cover the dataset exactly once")
    domain_validation = [index for indices in manifest["validation"]["domain_indices"].values() for index in indices]
    if sorted(domain_validation) != sorted(validation):
        raise ValueError("domain validation sets must match the global validation set")
    if manifest["schema_version"] == 2:
        held_out_clients = manifest["validation"]["clients"]
        if set(held_out_clients) != set(clients):
            raise ValueError("validation client IDs do not match training client IDs")
        client_validation = [
            index for entry in held_out_clients.values() for index in entry["indices"]
        ]
        if sorted(client_validation) != sorted(validation):
            raise ValueError("client validation sets must cover the holdout exactly once")
        for entry in held_out_clients.values():
            if sum(entry["domain_counts"].values()) != len(entry["indices"]):
                raise ValueError("incorrect client validation domain counts")
    for client in clients.values():
        if client["dataset_size"] != len(client["train_indices"]):
            raise ValueError("incorrect client dataset size")
        if sum(client["domain_counts"].values()) != client["dataset_size"]:
            raise ValueError("incorrect client domain counts")


def load_manifest(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a LoRACast partition manifest")
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--mode", choices=("iid", "mild", "strong"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split="train", revision=args.dataset_revision)
    manifest = build_manifest(
        dataset,
        dataset_name=args.dataset,
        dataset_revision=args.dataset_revision,
        num_clients=args.num_clients,
        mode=args.mode,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps({
        "manifest": str(args.output),
        "mode": args.mode,
        "dataset_sha256": manifest["dataset_sha256"],
        "client_domain_counts": {key: value["domain_counts"] for key, value in manifest["clients"].items()},
        "validation_domain_counts": manifest["validation"]["domain_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
