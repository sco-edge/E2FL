import psutil, socket, logging
from flwr.app import ArrayRecord, Context
from transformers import TrainingArguments

def cfg(ctx: Context, key: str, default=None):
    return ctx.run_config.get(key, default)

def lora_cfg_from(ctx: Context) -> dict:
    if not cfg(ctx, "lora-enabled", False):
        return {}
    targets = cfg(ctx, "lora-targets", "")
    return {
        "enabled": True,
        "r": int(cfg(ctx, "lora-r", 4)),
        "alpha": int(cfg(ctx, "lora-alpha", 8)),
        "dropout": float(cfg(ctx, "lora-dropout", 0.05)),
        "targets": [t.strip() for t in targets.split(",") if t.strip()],
        "task_type": cfg(ctx, "lora-task-type", "CAUSAL_LM"),
    }

def training_arguments_from(ctx: Context) -> TrainingArguments:
    """Build TrainingArguments from pyproject-style flat keys.

    Keys follow the pattern:
        training-arguments.output-dir
        training-arguments.learning-rate
        ...
    """
    # helper for key lookup
    def arg(key: str, default=None):
        return cfg(ctx, f"training-arguments.{key}", default)

    # Now build the HF TrainingArguments object
    return TrainingArguments(
        output_dir = arg("output-dir", ""),
        learning_rate = float(arg("learning-rate", 2e-4)) if arg("learning-rate") not in ["", None] else 2e-4,
        per_device_train_batch_size = int(arg("per-device-train-batch-size", 16)),
        gradient_accumulation_steps = int(arg("gradient-accumulation-steps", 1)),
        logging_steps = int(arg("logging-steps", 10)),
        num_train_epochs = float(arg("num-train-epochs", 3)),
        max_steps = int(arg("max-steps", -1)) if arg("max-steps") not in ["", None] else -1,
        save_steps = int(arg("save-steps", 1000)),
        save_total_limit = int(arg("save-total-limit", 10)),
        gradient_checkpointing = bool(arg("gradient-checkpointing", True)),
        lr_scheduler_type = arg("lr-scheduler-type", "constant"),
        remove_unused_columns = bool(arg("remove-unused-columns", True)),               # LLM fine-tuning 필수
        ddp_find_unused_parameters = bool(arg("ddp-find-unused-parameters", True)),          # LLM fine-tuning 필수
        bf16 = bool(arg("bf16", True)),                                # 필요 시 configurable
        fp16 = bool(arg("fp16", True)),
        report_to = arg("report-to","none"),                           # WandB 등 비활성화
    )

def get_last_octet_from_ip(interface: str | None = None) -> int:
    """주어진 인터페이스(또는 우선순위 인터페이스들)의 non-loopback IPv4 마지막 옥텟을 반환.
    없으면 ValueError를 발생시킴."""
    def _ip_for_iface(iface: str) -> str | None:
        addrs = psutil.net_if_addrs().get(iface, [])
        for a in addrs:
            if getattr(a, "family", None) == socket.AF_INET:
                addr = getattr(a, "address", None)
                if addr and not addr.startswith("127."):
                    return addr
        return None

    # 1) 명시된 인터페이스 우선
    if interface:
        ip = _ip_for_iface(interface)
        if ip:
            return int(ip.split(".")[-1])

    # 2) 우선순위 검사 (wlP1p1s0, wlan0 순서)
    preferred = ["wlP1p1s0", "wlan0"]
    for iface in preferred:
        ip = _ip_for_iface(iface)
        if ip:
            return int(ip.split(".")[-1])

    # 3) 시스템의 첫 non-loopback IPv4 사용
    for iface, addrs in psutil.net_if_addrs().items():
        for a in addrs:
            if getattr(a, "family", None) == socket.AF_INET:
                addr = getattr(a, "address", None)
                if addr and not addr.startswith("127."):
                    return int(addr.split(".")[-1])

    # 4) 실패
    raise ValueError("No non-loopback IPv4 address found on preferred interfaces")

def parse_node_config_string(s: str) -> dict:
    """문자열 형태 node-config (예: 'partition-id=0 num-partitions=2') 파싱하여 dict 반환."""
    out = {}
    if not s:
        return out
    # 허용되는 구분: 공백으로 구분된 key=value 쌍
    for token in s.split():
        if "=" in token:
            k, v = token.split("=", 1)
            k = k.strip()
            v = v.strip()
            # 숫자면 int로 변환 시도
            try:
                out[k] = int(v)
            except Exception:
                out[k] = v
    return out

def get_network_interface():
    interfaces = psutil.net_if_addrs().keys()
    return list(interfaces)

def get_device_name():
    interface_list = get_network_interface()
    if 'wlan0' in interface_list and 'wlp1s0' in interface_list:
        iface = 'wlp1s0' if validate_network_interface('wlp1s0') else "wlan0"
        return iface, 'RPi5', "PMIC"
    elif 'wlP1p1s0' in interface_list:
        iface = 'wlP1p1s0'
        return iface, 'jetson', "INA3221"
    else:
        return 'wlan0', 'RPi3', "Monsoon"

def get_network_usage(interface):
    """Get the current network usage for the specified interface."""
    net_io = psutil.net_io_counters(pernic=True)
    if interface in net_io:
        return {"bytes_sent": net_io[interface].bytes_sent, "bytes_recv": net_io[interface].bytes_recv}
    else:
        raise ValueError(f"Interface {interface} not found.")

def validate_network_interface(interface):
    """
    Validate if the given network interface exists on the system.
    :param interface: Network interface name to validate.
    :return: True if the interface exists, False otherwise.
    """
    if interface in psutil.net_if_addrs():
        return True
    else:
        logging.error(f"Invalid network interface: {interface}")
        return False

def get_network_interface():
    interfaces = psutil.net_if_addrs().keys()
    return list(interfaces)
