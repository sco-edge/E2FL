import math
import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from peft.utils import prepare_model_for_kbit_training
from e2fl.task import is_hf_model_name

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))

def is_llm_model_name(model_name: str) -> bool:
    """Return True if the model name looks like a small LLM used with LoRA.

    This helper normalizes optional 'hf:' prefixes and then checks for
    common small-LLM families such as TinyLlama, Phi, Mistral, Gemma,
    LLaMA, and Qwen.
    """
    name = model_name.lower()
    if name.startswith("hf:"):
        name = name[3:]
    return any(x in name for x in ["tinyllama", "phi", "mistral", "gemma", "llama", "qwen"])

def get_model(model_name: str, num_classes: int = None, dataset_name: str = None,
              lora_cfg: dict | None = None, gradient_checkpointing: bool = False, quantization: int | None = None):
    """Return a torch.nn.Module.

    - If the model name corresponds to a small LLM (e.g., Gemma3 270M, TinyLlama,
      Phi, Mistral, LLaMA, Qwen), we construct an AutoModelForCausalLM and wrap
      it with a LoRA adapter using the provided lora_cfg.
    - If the model name is a generic HF identifier (e.g., 'hf:bert-base-uncased'),
      we return an AutoModelForSequenceClassification.
    - Otherwise, we fall back to torchvision CNN architectures.
    """
    # Normalize HF-style model names
    hf_name = model_name[3:] if model_name.startswith("hf:") else model_name

    # 1) LLM + LoRA path (Gemma, TinyLlama, Phi, Mistral, LLaMA, Qwen, etc.)
    if is_llm_model_name(model_name):
        # Configure optional k-bit quantization for LLMs
        quantization_config = None
        # Treat quantization=0 or None as "no quantization"
        if quantization in (4, 8):
            if quantization == 4:
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == 8:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization not in (None, 0):
            # Guard against unsupported values
            raise ValueError(
                f"Use 0 (no quantization), 4-bit or 8-bit quantization. You passed: {quantization}"
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            quantization_config=quantization_config,
            device_map="auto" if quantization_config is not None else None,
            torch_dtype=None,
        )

        if gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
            # If desired, uncomment the following to prepare for k-bit training:
            # base_model = prepare_model_for_kbit_training(
            #     base_model, use_gradient_checkpointing=True
            # )

        # Freeze base model for adapter fine-tuning (FlowerTune-style)
        for param in base_model.parameters():
            param.requires_grad = False

        # Normalize and default LoRA configuration
        if lora_cfg is None:
            lora_cfg = {}

        # Rank (r)
        r = int(
            lora_cfg.get("r")
            or lora_cfg.get("rank")
            or lora_cfg.get("lora-r")
            or lora_cfg.get("lora_r")
            or 16
        )
        # Scaling (alpha)
        alpha = int(
            lora_cfg.get("alpha")
            or lora_cfg.get("lora_alpha")
            or lora_cfg.get("lora-alpha")
            or 32
        )
        # Dropout
        dropout = float(
            lora_cfg.get("dropout")
            or lora_cfg.get("lora_dropout")
            or lora_cfg.get("lora-dropout")
            or 0.05
        )
        # Target modules: accept both list and comma-separated string
        targets = (
            lora_cfg.get("targets")
            or lora_cfg.get("target_modules")
            or lora_cfg.get("lora-target-modules")
            or "q_proj,k_proj,v_proj,o_proj"
        )
        if isinstance(targets, str):
            targets = [t.strip() for t in targets.split(",") if t.strip()]

        task_type = lora_cfg.get("task_type", "CAUSAL_LM")

        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=targets,
            lora_dropout=dropout,
            task_type=task_type,
        )

        return get_peft_model(base_model, peft_config)

    """Return a torch.nn.Module. Supports torchvision models and HF transformers.
    For HF models, prefix model_name with 'hf:' or pass a HF model identifier (e.g., 'bert-base-uncased')."""
    # 2) HF sequence classification models (non-LLM)
    if is_hf_model_name(model_name) and not is_llm_model_name(model_name):
        return AutoModelForSequenceClassification.from_pretrained(hf_name, num_labels=num_classes)

    # MNIST 등 1채널 입력이 필요한 데이터셋인지 확인
    is_gray = dataset_name in ["mnist", "fashion_mnist"]

    if model_name == "resnet18":
        model = models.resnet18(num_classes=num_classes)
        if is_gray:
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif model_name == "resnet50":
        model = models.resnet50(num_classes=num_classes)
        if is_gray:
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif model_name == "resnext50":
        return models.resnext50_32x4d(num_classes=num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(num_classes=num_classes)
        if is_gray:
            model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        return model
    elif model_name == "alexnet":
        model = models.alexnet(num_classes=num_classes)
        if is_gray:
            model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2, bias=False)
        return model
    elif model_name == "convnext_tiny":
        return models.convnext_tiny(num_classes=num_classes)
    elif model_name == "squeezenet1":
        return models.squeezenet1_0(num_classes=num_classes)
    elif model_name == "densenet121":
        return models.densenet121(num_classes=num_classes)
    elif model_name == "densenet161":
        return models.densenet161(num_classes=num_classes)
    elif model_name == "inception_v3":
        return models.inception_v3(num_classes=num_classes, aux_logits=False)
    elif model_name == "googlenet":
        return models.googlenet(num_classes=num_classes, aux_logits=False)
    elif model_name == "shufflenet_v2":
        return models.shufflenet_v2_x1_0(num_classes=num_classes)
    elif model_name == "mobilenet_v2":
        return models.mobilenet_v2(num_classes=num_classes)
    elif model_name == "mobilenet_v3_small":
        return models.mobilenet_v3_small(num_classes=num_classes)
    elif model_name == "mnasnet1":
        return models.mnasnet1_0(num_classes=num_classes)
    elif model_name == "lenet":
        return Net()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
