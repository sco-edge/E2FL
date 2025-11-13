import numpy as np
from PIL import Image
import torch, logging, os, io
from typing import Optional
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from datasets.utils.logging import disable_progress_bar
from e2fl.task import is_hf_model_name

# 디버그 토글: E2FL_DEBUG=1 로 디버그 활성화 (기본 비활성)
E2FL_DEBUG = os.environ.get("E2FL_DEBUG", "0") == "1"
# 기존 DEBUG 변수를 대체하여 환경변수 기반으로 동작하게 함
DEBUG = E2FL_DEBUG

# task 전용 로거
logger = logging.getLogger("e2fl_dataset")
if not logger.handlers:
    logger.setLevel(logging.DEBUG if E2FL_DEBUG else logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

disable_progress_bar()
fds = None  # Cache FederatedDataset

def get_num_classes(dataset_name: str):
    if dataset_name in ["cifar10", "mnist", "fashion_mnist"]:
        return 10
    elif dataset_name == "cifar100":
        return 100
    elif dataset_name == "imagenet":
        return 1000
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

# https://github.com/adap/flower/blob/main/examples/flowertune-llm/flowertune_llm/dataset.py
def formatting_prompts_func(example):
    output_texts = []
    # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["instruction"])):
        text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Response: {example['response'][i]}"
        output_texts.append(text)
    return output_texts

def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict

def _unwrap_collate(batch):
    """Collate function handling dataset items that may already contain batched images.
    Expects `batch` usually of length 1 when dataset returns batched samples.
    Returns {'img': Tensor[B,C,H,W], 'label': Tensor[B]}."""
    if not batch:
        return {}
    item = batch[0]
    if isinstance(item, dict) and "img" in item:
        imgs = item["img"]

        # --- normalize imgs -> Tensor with shape (B,C,H,W) or list handled below ---
        if isinstance(imgs, list):
            imgs_t = []
            for x in imgs:
                if isinstance(x, torch.Tensor):
                    imgs_t.append(x)
                else:
                    arr = np.asarray(x)
                    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                        arr = np.transpose(arr, (2, 0, 1))
                    imgs_t.append(torch.tensor(arr))
            imgs = torch.stack(imgs_t)
        elif isinstance(imgs, np.ndarray):
            arr = imgs
            if arr.ndim == 4 and arr.shape[-1] in (1, 3):
                imgs = torch.tensor(arr).permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W
            else:
                imgs = torch.tensor(arr)
        elif isinstance(imgs, torch.Tensor):
            if imgs.ndim == 4 and imgs.shape[-1] in (1, 3):
                imgs = imgs.permute(0, 3, 1, 2)

        # Ensure batch dim exists
        if isinstance(imgs, torch.Tensor):
            if imgs.ndim == 3:
                logger.debug("[_unwrap_collate] imgs is 3D tensor, treating as single image CHW -> BCHW")
                imgs = imgs.unsqueeze(0)
            elif imgs.ndim == 2:
                imgs = imgs.unsqueeze(0).unsqueeze(0)

        B = imgs.shape[0] if isinstance(imgs, torch.Tensor) else (len(imgs) if isinstance(imgs, (list, np.ndarray)) else 1)

        # --- Robust label lookup: prefer explicit presence of keys (avoid falsy-or issues) ---
        labels = None
        if "label" in item:
            labels = item["label"]
        elif "labels" in item:
            labels = item["labels"]
        elif "label_ids" in item:
            labels = item["label_ids"]
        elif "y" in item:
            labels = item["y"]
        elif "target" in item:
            labels = item["target"]

        # Treat empty-list as missing
        if isinstance(labels, (list, np.ndarray)) and len(labels) == 0:
            logger.debug("[_unwrap_collate] found empty label list -> treating as missing")
            labels = None

        # If still missing, debug print
        if labels is None:
            if E2FL_DEBUG:
                try:
                    logger.debug(f"[ _unwrap_collate ] item keys: {list(item.keys())}")
                    if "img" in item:
                        try:
                            tmp = np.asarray(item["img"])
                            logger.debug(f"[ _unwrap_collate ] raw img type={type(item['img'])}, shape={getattr(tmp,'shape',None)}, dtype={getattr(tmp,'dtype',None)}")
                        except Exception:
                            logger.debug("[ _unwrap_collate ] cannot preview img")
                    logger.debug("[ _unwrap_collate ] labels field missing or None -> will create dummy labels")
                except Exception:
                    logger.debug("[ _unwrap_collate ] failed to emit debug info for item")

            labels_t = torch.full((B,), -1, dtype=torch.long)
        else:
            # Normalize labels into tensor
            if isinstance(labels, torch.Tensor):
                labels_t = labels
            elif isinstance(labels, (list, np.ndarray)):
                labels_t = torch.tensor(labels, dtype=torch.long)
            else:
                labels_t = torch.tensor([labels], dtype=torch.long)

            # Ensure labels length matches B
            if labels_t.numel() != B:
                if labels_t.numel() == 1:
                    labels_t = labels_t.expand(B)
                    logger.debug(f"[ _unwrap_collate ] Expanded scalar label to length {B}")
                else:
                    logger.debug(f"[ _unwrap_collate ] label size mismatch (labels={labels_t.shape}, B={B}), creating dummy labels")
                    labels_t = torch.full((B,), -1, dtype=torch.long)

        return {"img": imgs, "label": labels_t}

    # fallback
    return _collate_vision(batch)

def _collate_vision(batch):
    # batch: list of samples where each sample is a dict with 'img' (tensor) and 'label' (int)
    imgs = torch.stack([sample["img"] for sample in batch])
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
    return {"img": imgs, "label": labels}

def is_vision_dataset(dataset_name: str) -> bool:
    vision_datasets = {"cifar10", "cifar100", "mnist", "fashion_mnist", "imagenet"}
    return dataset_name in vision_datasets

def is_text_dataset(dataset_name: str) -> bool:
    text_datasets = {"imdb", "ag_news", "yelp_review", "sst2"}  # 필요시 확장
    return dataset_name in text_datasets

def is_llm_model_name(name: str):
    if not name:
        return False
    name = name.lower()
    return any(x in name for x in ["tinyllama", "phi", "mistral", "gemma", "llama", "qwen"])

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


def load_data(dataset_name: str, partition_id: int, num_partitions: int, 
              batch_size: int, model_name: Optional[str] = None, 
              seq_len: Optional[int] = None, mode: str = "train",
              task_type: str = "CAUSAL_LM"):
    """
    Return (trainloader, testloader).
    For vision datasets: batches are dicts {'img': Tensor, 'label': Tensor}.
    For text/HF datasets: batches are dicts suitable for HF models (input_ids, attention_mask, labels...)
    mode: "train" | "eval"
    task_type: "CAUSAL_LM" | "SEQ_CLS"
    """
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )

    # 안전하게 partition_id 로드: out-of-range 예외를 잡아 모듈로로 보정
    try:
        partition = fds.load_partition(partition_id)
    except Exception as e:
        logger.warning(f"load_partition failed for partition_id={partition_id}: {e}. Attempting fallback.")
        # 시도: fds가 제공하는 전체 샤드/파티션 개수를 얻어 모듈로로 보정
        safe_id = 0
        try:
            num_shards = getattr(fds, "num_partitions", None) or getattr(fds, "num_shards", None)
            if num_shards and isinstance(num_shards, int) and num_shards > 0:
                safe_id = int(partition_id) % int(num_shards)
        except Exception:
            safe_id = 0
        logger.warning(f"Using fallback partition_id={safe_id}.")
        try:
            partition = fds.load_partition(safe_id)
        except Exception as e2:
            raise RuntimeError(f"Failed to load partition (original={partition_id}, fallback={safe_id}): {e2}") from e2
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    if is_vision_dataset(dataset_name):
        transforms = get_transforms(dataset_name)

        def apply_transforms(sample):
            # 지원되는 key 탐색
            img = sample.get("img") or sample.get("image") or sample.get("pixels")
            if img is None:
                return sample

            if DEBUG:
                try:
                    arr_preview = np.asarray(img)
                    print(f"[DEBUG apply_transforms] img type={type(img)}, arr.shape={arr_preview.shape}, dtype={arr_preview.dtype}")
                except Exception:
                    print(f"[DEBUG apply_transforms] img type={type(img)}, cannot convert to ndarray for preview")

            def to_pil_safe(x):
                """Return a PIL.Image or a list of PIL.Image when x is batched (4D)."""
                arr = np.asarray(x, dtype=np.uint8)
                if DEBUG:
                    print(f"[DEBUG to_pil_safe] initial arr.shape={getattr(arr,'shape',None)}, dtype={getattr(arr,'dtype',None)}")

                # If this is a batch: handle per-sample conversion and return list of PIL images
                if arr.ndim == 4:
                    if DEBUG:
                        print(f"[DEBUG to_pil_safe] detected 4D batch, B={arr.shape[0]}")
                    pil_list = []
                    for i in range(arr.shape[0]):
                        sub = arr[i]
                        # try squeeze/transposes and typical candidates
                        candidates = [sub]
                        try:
                            candidates.append(np.squeeze(sub))
                        except Exception:
                            pass
                        # if channel-first
                        try:
                            if sub.ndim == 3 and sub.shape[0] in (1, 3) and sub.shape[-1] not in (1, 3):
                                candidates.append(np.transpose(sub, (1, 2, 0)))
                        except Exception:
                            pass
                        converted = None
                        for cand in candidates:
                            try:
                                converted = Image.fromarray(cand)
                                break
                            except Exception:
                                continue
                        if converted is None:
                            shapes = [getattr(c, "shape", None) for c in candidates]
                            raise TypeError(f"Cannot convert batch item to PIL Image, tried shapes: {shapes}")
                        pil_list.append(converted)
                    return pil_list

                # Non-batched: try multiple candidates
                candidates = []
                candidates.append(arr)
                try:
                    candidates.append(np.squeeze(arr))
                except Exception:
                    pass

                a = arr
                while a.ndim > 2 and a.shape[0] == 1:
                    a = a[0]
                    candidates.append(a)

                total = arr.size
                if total == 32 * 32 * 3:
                    candidates.append(arr.reshape(32, 32, 3))
                if total == 32 * 32:
                    candidates.append(arr.reshape(32, 32))

                for cand in list(candidates):
                    try:
                        if cand.ndim == 3 and cand.shape[0] in (1, 3) and cand.shape[-1] not in (1, 3):
                            candidates.append(np.transpose(cand, (1, 2, 0)))
                    except Exception:
                        pass

                for cand in candidates:
                    try:
                        return Image.fromarray(cand)
                    except Exception:
                        continue

                shapes = [getattr(c, "shape", None) for c in candidates]
                raise TypeError(f"Cannot convert to PIL Image, tried candidate shapes: {shapes}")

            # bytes 처리(이미지 바이트라면)
            if isinstance(img, (bytes, bytearray)):
                try:
                    pil_img = Image.open(io.BytesIO(img)).convert("RGB")
                except Exception:
                    pil_img = to_pil_safe(img)
            elif isinstance(img, Image.Image):
                pil_img = img
            else:
                pil_img = to_pil_safe(img)

            # pil_img can be a single PIL.Image or a list of PIL.Image (if dataset returned batch)
            if isinstance(pil_img, list):
                # apply transforms per image -> produce list of tensors
                transformed_imgs = [transforms(p) for p in pil_img]
                sample["img"] = transformed_imgs
                # normalize label: if vector-like keep, else broadcast if scalar
                if "label" in sample:
                    lbl = sample["label"]
                    if isinstance(lbl, (list, np.ndarray)) and len(lbl) == len(transformed_imgs):
                        sample["label"] = lbl
                    else:
                        sample["label"] = [lbl] * len(transformed_imgs)
            else:
                sample["img"] = transforms(pil_img)

            # label 키 일관성 유지
            if "label" not in sample and "labels" in sample:
                sample["label"] = sample["labels"]
            return sample

        partition_train_test = partition_train_test.with_transform(apply_transforms)

        # Use batch_size=1 to avoid double-batching when dataset already returns batched items,
        # and use _unwrap_collate to standardize outputs.
        trainloader = DataLoader(partition_train_test["train"], batch_size=1, shuffle=True, collate_fn=_unwrap_collate)
        testloader = DataLoader(partition_train_test["test"], batch_size=1, collate_fn=_unwrap_collate)
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
    
    elif is_llm_model_name(model_name):
        hf_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=True)

        def tokenize_fn(sample):
            text = sample.get("text") or sample.get("sentence") or sample.get("review") or sample.get("input")
            enc = tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=seq_len or 1024,
            )
            # causal LM label = shifted input
            enc["labels"] = enc["input_ids"].copy()
            return enc

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # LLM은 causal LM, MLM 아님
        )

        partition_train_test = partition_train_test.with_transform(tokenize_fn)

        trainloader = DataLoader(
            partition_train_test["train"],
            batch_size=batch_size,
            shuffle=(mode == "train"),
            collate_fn=collator,
        )
        testloader = DataLoader(
            partition_train_test["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )
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

