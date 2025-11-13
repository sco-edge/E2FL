"""E2FL: A Flower / PyTorch app."""
from collections import OrderedDict
import numpy as np
from PIL import Image
import io
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision.models as models
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

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

def is_vision_dataset(dataset_name: str) -> bool:
    vision_datasets = {"cifar10", "cifar100", "mnist", "fashion_mnist", "imagenet"}
    return dataset_name in vision_datasets

def is_text_dataset(dataset_name: str) -> bool:
    text_datasets = {"imdb", "ag_news", "yelp_review", "sst2"}  # 필요시 확장
    return dataset_name in text_datasets

def is_hf_model_name(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    if model_name.startswith("hf:"):
        return True
    hf_indicators = ["bert", "roberta", "distilbert", "gpt2", "deberta", "electra", "xlnet"]
    return any(ind in model_name.lower() for ind in hf_indicators)

def get_model(model_name: str, num_classes: int, dataset_name: str = None):
    """Return a torch.nn.Module. Supports torchvision models and HF transformers.
    For HF models, prefix model_name with 'hf:' or pass a HF model identifier (e.g., 'bert-base-uncased')."""
    if is_hf_model_name(model_name):
        hf_name = model_name[3:] if model_name.startswith("hf:") else model_name
        # Use AutoModelForSequenceClassification for text models
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

def get_num_classes(dataset_name: str):
    if dataset_name in ["cifar10", "mnist", "fashion_mnist"]:
        return 10
    elif dataset_name == "cifar100":
        return 100
    elif dataset_name == "imagenet":
        return 1000
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_transforms(dataset_name: str):
    if is_vision_dataset(dataset_name):
        if dataset_name in ["cifar10", "cifar100"]:
            return Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif dataset_name in ["mnist", "fashion_mnist"]:
            return Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        else:
            return Compose([ToTensor()])
    # 텍스트의 경우 transform은 tokenizer로 처리
    return None

def _collate_vision(batch):
    # batch: list of samples where each sample is a dict with 'img' (tensor) and 'label' (int)
    imgs = torch.stack([sample["img"] for sample in batch])
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
    return {"img": imgs, "label": labels}

def load_data(dataset_name: str, partition_id: int, num_partitions: int, batch_size: int, model_name: Optional[str] = None):
    """Return (trainloader, testloader).
    For vision datasets: batches are dicts {'img': Tensor, 'label': Tensor}.
    For text/HF datasets: batches are dicts suitable for HF models (input_ids, attention_mask, labels...)."""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    if is_vision_dataset(dataset_name):
        transforms = get_transforms(dataset_name)

        def apply_transforms(sample):
            sample["img"] = transforms(sample["img"])
            # ensure label key is 'label'
            if "label" not in sample and "labels" in sample:
                sample["label"] = sample["labels"]
            return sample

        partition_train_test = partition_train_test.with_transform(apply_transforms)

        trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True, collate_fn=_collate_vision)
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size, collate_fn=_collate_vision)
        return trainloader, testloader

    elif is_text_dataset(dataset_name) or is_hf_model_name(model_name):
        # Use HF tokenizer + DataCollatorWithPadding for text datasets / HF models
        hf_model = model_name[3:] if (model_name and model_name.startswith("hf:")) else model_name or "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
        data_collator = DataCollatorWithPadding(tokenizer)

        # Expect partition dataset to provide raw text under key 'text' and label under 'label' or 'labels'
        def tokenize_fn(sample):
            text = sample.get("text") or sample.get("sentence") or sample.get("review") or sample.get("input")
            enc = tokenizer(text, truncation=True, padding=False)
            # keep label
            if "label" in sample:
                enc["labels"] = sample["label"]
            elif "labels" in sample:
                enc["labels"] = sample["labels"]
            return enc

        partition_train_test = partition_train_test.with_transform(tokenize_fn)

        # DataLoader must not apply default collate; use HF data_collator
        trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size, collate_fn=data_collator)
        return trainloader, testloader

    else:
        # Fallback: treat as vision-like using default transforms
        transforms = get_transforms(dataset_name)
        def apply_transforms(sample):
            if "img" in sample:
                sample["img"] = transforms(sample["img"]) if transforms else sample["img"]
            return sample
        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True, collate_fn=_collate_vision)
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size, collate_fn=_collate_vision)
        return trainloader, testloader

'''
LLM Fine-tuning with Flower [https://github.com/adap/flower/tree/main/examples/flowertune-llm]
Vision Transformer with Flower [https://github.com/adap/flower/tree/main/examples/flowertune-vit]
'''

disable_progress_bar()
fds = None  # Cache FederatedDataset

def train(net, trainloader, epochs, device) -> None:
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-5)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device) -> tuple[Any | float, Any]:
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
