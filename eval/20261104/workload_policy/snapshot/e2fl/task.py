"""E2FL: A Flower / PyTorch app."""
from collections import OrderedDict
import numpy as np
from typing import Any, Optional
import logging, os, torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from fvcore.nn import FlopCountAnalysis

# 디버그 토글: E2FL_DEBUG=1 로 디버그 활성화 (기본 비활성)
E2FL_DEBUG = os.environ.get("E2FL_DEBUG", "0") == "1"
# 기존 DEBUG 변수를 대체하여 환경변수 기반으로 동작하게 함
DEBUG = E2FL_DEBUG

# task 전용 로거
logger = logging.getLogger("e2fl_task")
if not logger.handlers:
    logger.setLevel(logging.DEBUG if E2FL_DEBUG else logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

def is_hf_model_name(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    if model_name.startswith("hf:"):
        return True
    hf_indicators = ["bert", "roberta", "distilbert", "gpt2", "deberta", "electra", "xlnet"]
    return any(ind in model_name.lower() for ind in hf_indicators)

def is_lora_model(model) -> bool:
    return hasattr(model, "peft_config") or any("lora" in n.lower() for n, _ in model.named_parameters())

def get_flops(net, device) -> int:
    dummy = torch.randn(1, 3, 224, 224).to(device)
    return FlopCountAnalysis(net, dummy).total()
    
def train(net, trainloader, epochs, device) -> float:
    """
    Unified train function:
    - Vision models: use CrossEntropyLoss(img/label)
    - LLM models (Causal LM): use HF forward loss
    """
    net.to(device)
    net.train()

    optimizer = torch.optim.Adam(
        (p for p in net.parameters() if p.requires_grad),
        lr=5e-5,
    )

    is_llm = hasattr(net, "base_model") or hasattr(net, "model") or "lm_head" in net.state_dict()

    if not is_llm:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    running_loss = 0.0
    total_batches = 0

    for _ in range(epochs):
        for batch in trainloader:

            optimizer.zero_grad()

            # --------------------------
            # 📌 LLM TRAINING FLOW
            # --------------------------
            if is_llm:
                # batch: input_ids / attention_mask / labels
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                labels = batch["labels"].to(device)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                outputs = net(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            # --------------------------
            # 📌 VISION TRAINING FLOW
            # --------------------------
            else:
                images = batch["img"]
                labels = batch["label"]

                if isinstance(images, list):
                    images = torch.stack([
                        x if isinstance(x, torch.Tensor) else torch.tensor(x)
                        for x in images
                    ])

                if not isinstance(images, torch.Tensor):
                    images = torch.tensor(images)

                if images.dim() == 3:
                    images = images.unsqueeze(0)

                images = images.to(device)
                labels = labels.to(device)

                bn_toggled = False
                if images.size(0) == 1:
                    bn_toggled = True
                    for m in net.modules():
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                            m.eval()

                outputs = net(images)
                loss = criterion(outputs, labels)

                if bn_toggled:
                    for m in net.modules():
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                            m.train()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

    avg_trainloss = running_loss / max(total_batches, 1)
    return avg_trainloss


def test(net, testloader, device) -> tuple[Any | float, Any]:
    """Validate the model on the test set."""
    net.to(device)
    net.eval()  # 평가 모드: BatchNorm/Dropout을 평가용 동작으로 변경
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"]
            # if labels might be None or -1 dummy
            if labels is None:
                logger.debug("[test] received None labels; skipping batch")
                continue
            labels = labels.to(device)
            # create mask to ignore dummy labels (-1)
            valid_mask = labels >= 0
            if valid_mask.sum().item() == 0:
                logger.debug("[test] no valid labels in batch; skipping")
                continue
            # select valid samples
            if valid_mask.all():
                imgs_sel = images
                labels_sel = labels
            else:
                imgs_sel = images[valid_mask]
                labels_sel = labels[valid_mask]
            outputs = net(imgs_sel)
            loss += criterion(outputs, labels_sel).item() * imgs_sel.size(0)
            correct += (torch.argmax(outputs, dim=1) == labels_sel).sum().item()
            total += imgs_sel.size(0)
    avg_loss = loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
        
def set_weights(net, parameters):
    """
    Robust set_weights:
    - 다양한 flwr Array wrapper 형태(.as_numpy/.to_numpy/.arrays/.value/.data/...)를 시도해서 numpy->tensor로 변환
    - bytes/memoryview -> frombuffer 처리 (대상 shape을 이용해 reshape 시도)
    - iterable -> np.array(list(...)) 시도
    - 실패 시 기존 파라미터를 fallback으로 사용
    - 첫 번째 반복 실패 항목에 대해 타입/dir/간단 repr 출력 (디버깅용)
    """
    if is_lora_model(net):
        try:
            sd_keys = list(get_peft_model_state_dict(net).keys())
            if isinstance(parameters, dict): # parameters가 dict일 경우: key matching 우선
                new_sd = {k: torch.tensor(v).cpu() for k, v in parameters.items() if k in sd_keys}
            else: # list/array case
                if len(parameters) != len(sd_keys):
                    logger.warning(f"[set_weights] LoRA adapter: key/weight count mismatch {len(sd_keys)} vs {len(parameters)})")
                new_sd = {k: torch.tensor(w).cpu() for k, w in zip(sd_keys, parameters)}

            set_peft_model_state_dict(net, new_sd)
            logger.debug(f"[set_weights] Loaded {len(new_sd)} LoRA adapter weights.")
            return  # ✅ LoRA branch handled, skip rest
        except Exception as e:
            logger.warning(f"[set_weights] LoRA adapter load failed: {e}. Falling back to default path.")

    state_keys = list(net.state_dict().keys())

    if isinstance(parameters, dict) or hasattr(parameters, "items"):
        items = parameters.items()
    else:
        try:
            params_list = list(parameters)
        except Exception as e:
            raise TypeError(f"Unsupported parameters format: {type(parameters)}") from e
        items = zip(state_keys, params_list)

    new_state = OrderedDict()
    first_fail_debugged = False

    # target shapes for reshape attempts
    target_shapes = {k: v.cpu().numpy().shape for k, v in net.state_dict().items()}

    for k, v in items:
        try:
            # 이미 tensor이면 그대로 사용
            if isinstance(v, torch.Tensor):
                new_state[k] = v.cpu()
                continue

            arr = None

            # 1) 시도: common numpy-like accessors
            for acc in ("as_numpy", "to_numpy", "numpy", "to_ndarray"):
                if hasattr(v, acc):
                    try:
                        fn = getattr(v, acc)
                        arr = fn() if callable(fn) else fn
                        break
                    except Exception:
                        arr = None

            # 2) 시도: common attrs
            if arr is None:
                for attr in ("arrays", "value", "data", "array", "buffer"):
                    if hasattr(v, attr):
                        try:
                            arr = getattr(v, attr)
                            break
                        except Exception:
                            arr = None

            # 3) tolist/tobytes/tobytearray
            if arr is None and hasattr(v, "tolist"):
                try:
                    arr = v.tolist()
                except Exception:
                    arr = None

            # 4) raw bytes / memoryview / tobytes
            if arr is None and isinstance(v, (bytes, bytearray, memoryview)):
                try:
                    arr = np.frombuffer(v, dtype=np.float32)
                except Exception:
                    arr = None

            if arr is None:
                # some wrappers provide .tobytes()
                if hasattr(v, "tobytes") and callable(getattr(v, "tobytes")):
                    try:
                        b = v.tobytes()
                        arr = np.frombuffer(b, dtype=np.float32)
                    except Exception:
                        arr = None

            # 5) __array__ protocol
            if arr is None:
                try:
                    arr = np.asarray(v)
                    # if result is object-dtype or scalar wrapper, treat arr further below
                except Exception:
                    arr = None

            # 6) iterable fallback: list(...) -> array
            if arr is None:
                try:
                    if hasattr(v, "__iter__") and not isinstance(v, (str, bytes, bytearray)):
                        lst = list(v)
                        # avoid huge unintended expansions: require at least one numeric element
                        if len(lst) > 0:
                            arr = np.array(lst)
                except Exception:
                    arr = None

            # If arr exists but is not ndarray, try to convert
            if arr is not None and not isinstance(arr, np.ndarray):
                try:
                    arr = np.asarray(arr)
                except Exception:
                    pass

            # If arr is a 1D flat array from bytes, try reshape to target shape if available
            if isinstance(arr, np.ndarray):
                if arr.ndim == 1 and k in target_shapes:
                    tgt = target_shapes[k]
                    try:
                        if arr.size == np.prod(tgt):
                            arr = arr.reshape(tgt)
                        # if sizes mismatch but dtype convertible, leave as-is and rely on load_state_dict strict=False
                    except Exception:
                        pass

            # Finally convert to tensor
            if isinstance(arr, np.ndarray):
                # ensure float compatibility
                try:
                    # if dtype is object, try to cast to float32
                    if arr.dtype == object:
                        arr = arr.astype(np.float32)
                    new_state[k] = torch.from_numpy(arr).cpu()
                    continue
                except Exception:
                    # proceed to next fallback
                    pass

            # try torch.as_tensor directly as last numeric attempt
            try:
                new_state[k] = torch.as_tensor(v).cpu()
                continue
            except Exception:
                pass

            # If we reach here, conversion failed -> fallback to existing param
            raise RuntimeError("Could not convert parameter to tensor")

        except Exception as e:
            logger.debug(f"[set_weights] failed to convert param {k} (type={type(v)}): {e}. Using existing parameter as fallback.")
            if not first_fail_debugged and E2FL_DEBUG:
                try:
                    tname = getattr(v, "__class__", None)
                    logger.debug(f"[set_weights] FIRST_FAIL type: {tname}")
                    logger.debug(f"[set_weights] FIRST_FAIL dir: {str(dir(v))[:400]}")  # truncate
                    try:
                        s = repr(v)
                        if len(s) > 400:
                            s = s[:400] + "...(truncated)"
                        logger.debug(f"[set_weights] FIRST_FAIL repr: {s}")
                    except Exception:
                        logger.debug("[set_weights] FIRST_FAIL repr: <repr-failed>")
                except Exception:
                    pass
                first_fail_debugged = True
            new_state[k] = net.state_dict()[k].clone().cpu()

    # load into model (allow partial loads)
    try:
        net.load_state_dict(new_state, strict=False)
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG set_weights] load_state_dict failed: {e}. Falling back to strict=True load attempt.")
        net.load_state_dict(new_state, strict=True)
