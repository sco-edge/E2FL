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

def get_model(model_name: str, num_classes: int = None, dataset_name: str = None,
              lora_cfg: dict | None = None, gradient_checkpointing: bool = False, quantization: int | None = None):
    if any(x in model_name.lower() for x in ["tinyllama", "phi", "mistral", "gemma", "llama", "qwen"]):
        if quantization == 4:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif quantization == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(
                f"Use 4-bit or 8-bit quantization. You passed: {quantization}/"
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",  # 중요
            torch_dtype = None,
            #low_cpu_mem_usage=True,
            #torch_dtype=torch.bfloat16,
        )

        if gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
            '''
            base_model = prepare_model_for_kbit_training(
                base_model, use_gradient_checkpointing=gradient_checkpointing
            )
            '''
        for param in base_model.parameters():
            param.requires_grad = False  # freeze base model for adapter fine-tuning (FlowerTune-style)
            
        peft_config = LoraConfig(
            r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["targets"],
            lora_dropout=lora_cfg["dropout"],
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )

        return get_peft_model(base_model, peft_config)

    """Return a torch.nn.Module. Supports torchvision models and HF transformers.
    For HF models, prefix model_name with 'hf:' or pass a HF model identifier (e.g., 'bert-base-uncased')."""
    if is_hf_model_name(model_name):
        hf_name = model_name[3:] if model_name.startswith("hf:") else model_name
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
