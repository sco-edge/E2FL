"""LoRACast LLM model construction.

This module is a small integration layer around the public Transformers and
PEFT APIs.  It intentionally contains only the LLM/LoRA path used by LoRACast;
vision/tutorial model implementations are not part of the registration build.
"""

from __future__ import annotations

import os
from typing import Any

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


_LLM_FAMILIES = ("gemma", "qwen", "llama", "mistral", "phi", "tinyllama")


def normalize_model_name(model_name: str) -> str:
    if not model_name:
        raise ValueError("model_name must be non-empty")
    return model_name[3:] if model_name.startswith("hf:") else model_name


def is_llm_model_name(model_name: str) -> bool:
    name = normalize_model_name(model_name).lower()
    return any(family in name for family in _LLM_FAMILIES)


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return default


def get_model(
    model_name: str,
    num_classes: int | None = None,
    dataset_name: str | None = None,
    lora_cfg: dict[str, Any] | None = None,
    gradient_checkpointing: bool = False,
    quantization: int | None = None,
):
    """Create the causal-LM + LoRA model used by LoRACast.

    ``num_classes`` and ``dataset_name`` are accepted only for call-site
    compatibility with the older application code.
    """
    del num_classes, dataset_name

    if not is_llm_model_name(model_name):
        raise ValueError(
            f"LoRACast registration runtime supports LLMs only: {model_name!r}"
        )

    config = dict(lora_cfg or {})
    hf_name = normalize_model_name(model_name)

    if quantization not in (None, 0, 4, 8):
        raise ValueError("quantization must be 0, 4, 8, or None")

    quant_config = None
    if quantization == 4:
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == 8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        quantization_config=quant_config,
        device_map="auto" if quant_config is not None else None,
    )

    if gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False

    # Only adapter parameters should be trainable.
    for parameter in base_model.parameters():
        parameter.requires_grad = False

    rank_default = os.environ.get("LORA_R", "16")
    rank = int(_first(config, "r", "rank", "lora-r", "lora_r", default=rank_default))
    alpha = int(_first(config, "alpha", "lora_alpha", "lora-alpha", default=32))
    dropout = float(
        _first(config, "dropout", "lora_dropout", "lora-dropout", default=0.05)
    )

    targets = _first(
        config,
        "targets",
        "target_modules",
        "lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
    )
    if isinstance(targets, str):
        targets = [item.strip() for item in targets.split(",") if item.strip()]
    else:
        targets = list(targets)

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        task_type=str(config.get("task_type", "CAUSAL_LM")),
    )
    return get_peft_model(base_model, peft_config)
