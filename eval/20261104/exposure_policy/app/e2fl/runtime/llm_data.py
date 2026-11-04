"""LoRACast instruction-tuning data loader.

The implementation uses the public Flower Datasets, Hugging Face tokenizer,
and PyTorch DataLoader APIs.  It deliberately avoids the older Flower example
prompt helper and builds a simple LoRACast-owned instruction format.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from e2fl.runtime.partition_manifest import dataset_digest, load_manifest


_DATASETS: dict[tuple[str, int], FederatedDataset] = {}


def normalize_model_name(model_name: str) -> str:
    if not model_name:
        raise ValueError("model_name is required for LoRACast LLM data loading")
    return model_name[3:] if model_name.startswith("hf:") else model_name


def _dataset(dataset_name: str, num_partitions: int) -> FederatedDataset:
    key = (dataset_name, int(num_partitions))
    cached = _DATASETS.get(key)
    if cached is not None:
        return cached

    partitioner = IidPartitioner(num_partitions=int(num_partitions))
    value = FederatedDataset(
        dataset=dataset_name,
        partitioners={"train": partitioner},
    )
    _DATASETS[key] = value
    return value


def _batch_text(batch: dict) -> list[str]:
    instructions = batch.get("instruction")
    if instructions is None:
        raise ValueError(
            "LoRACast instruction tuning requires an 'instruction' column"
        )

    count = len(instructions)
    contexts = batch.get("input") or [""] * count
    responses = batch.get("output")
    if responses is None:
        responses = batch.get("response")
    if responses is None:
        raise ValueError(
            "LoRACast instruction tuning requires an 'output' or 'response' column"
        )

    texts: list[str] = []
    for instruction, context, response in zip(instructions, contexts, responses):
        pieces = ["Instruction:", str(instruction).strip()]
        context_text = str(context or "").strip()
        if context_text:
            pieces.extend(["Context:", context_text])
        pieces.extend(["Answer:", str(response).strip()])
        texts.append("\n".join(pieces))
    return texts


def load_data(
    dataset_name: str,
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    model_name: str,
    max_seq_len: int = 256,
    partition_manifest_path: str | None = None,
):
    """Return ``(trainloader, None)`` for LoRACast causal-LM fine-tuning."""
    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")

    if partition_manifest_path:
        # Experimental path: fail closed if the client has the wrong manifest or
        # dataset snapshot. The legacy IID partitioner remains the default.
        from datasets import load_dataset

        manifest_path = Path(partition_manifest_path)
        if not manifest_path.is_absolute():
            workspace_root = Path(os.environ.get(
                "LORACAST_WORKSPACE_ROOT", Path(__file__).resolve().parents[2]
            ))
            manifest_path = workspace_root / manifest_path
        manifest = load_manifest(manifest_path)
        if manifest["dataset_name"] != dataset_name:
            raise ValueError("partition manifest dataset does not match run config")
        if int(manifest["num_clients"]) != int(num_partitions):
            raise ValueError("partition manifest client count does not match node config")
        if not 0 <= int(partition_id) < num_partitions:
            raise ValueError("partition ID is outside the manifest client range")
        dataset = load_dataset(
            dataset_name, split="train", revision=manifest.get("dataset_revision")
        )
        if len(dataset) != manifest["dataset_size"] or dataset_digest(dataset) != manifest["dataset_sha256"]:
            raise ValueError("local dataset content differs from partition manifest")
        train_ds = dataset.select(manifest["clients"][str(partition_id)]["train_indices"])
    else:
        partition_id = int(partition_id) % int(num_partitions)
        fds = _dataset(dataset_name, num_partitions)
        train_ds = fds.load_partition(partition_id, "train")

    tokenizer = AutoTokenizer.from_pretrained(
        normalize_model_name(model_name),
        use_fast=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has neither pad_token nor eos_token; configure a pad token "
                "explicitly for this model."
            )
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            _batch_text(batch),
            truncation=True,
            max_length=int(max_seq_len),
            padding=False,
        )

    tokenized = train_ds.map(
        tokenize,
        batched=True,
        remove_columns=train_ds.column_names,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    loader = DataLoader(
        tokenized,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=collator,
    )
    return loader, None
