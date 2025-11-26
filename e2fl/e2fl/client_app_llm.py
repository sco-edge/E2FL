"""E2FL: A Flower / PyTorch app."""
from log import WrlsEnv
from power import _power_monitor_interface
from power.powermon import get_power_monitor

from datetime import datetime
import psutil, logging, time, socket, csv, importlib, logging, os

import torch
import flwr as fl
from flwr.clientapp import ClientApp
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from e2fl.task import set_weights, test, train, is_lora_model
from e2fl.models import get_model
from e2fl.dataset import load_data, get_tokenizer_and_data_collator_and_propt_formatting, get_num_classes
from e2fl.utils import cfg, lora_cfg_from, training_arguments_from, get_device_name, get_last_octet_from_ip, parse_node_config_string, get_network_usage
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from flowertune_llm.models import cosine_annealing
from transformers import TrainingArguments
from trl import SFTTrainer

class CustomFormatter(logging.Formatter):
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

##############################################################################################################

_LOCAL_PATH = './eval/'

# 전역 logger 설정 (CustomFormatter 사용)
logger = logging.getLogger("e2fl_client")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

##############################################################################################################

def get_partition_id_from_ip(context: Context | None = None):
    """
    우선 인터페이스( wlP1p1s0, wlan0 )에서 IPv4 마지막 옥텟을 가져옴.
    없으면 context.run_config의 node-config (예: "partition-id=0 num-partitions=2")를 파싱하여 사용.
    반환: (partition_id:int, num_partitions:int)
    """
    # 1) 시도: 실제 인터페이스에서 last octet 읽기
    try:
        # 사용 중인 인터페이스가 _LOCAL_STATE에 저장되어 있을 경우 우선 사용
        iface = _LOCAL_STATE.get("interfaces") if isinstance(_LOCAL_STATE, dict) else None
        last = None
        try:
            last = get_last_octet_from_ip(iface) if iface else get_last_octet_from_ip()
        except Exception:
            # 무시하고 node-config로 폴백
            last = None

        if last is not None:
            # node-config에서 num-partitions 정보가 있으면 사용, 없으면 1로 설정
            num_parts = 1
            if context is not None:
                nc = context.run_config.get("node-config") or context.run_config.get("node_config") or context.run_config.get("node_config_str")
                if isinstance(nc, str):
                    parsed = parse_node_config_string(nc)
                    num_parts = int(parsed.get("num-partitions", parsed.get("num_partitions", num_parts)))
            partition_id = int(last) % num_parts
            return partition_id, int(num_parts)
    except Exception:
        # 의도적 무시 — 아래 node-config 처리로 넘어감
        pass

    # 2) node-config에서 partition-id/num-partitions 파싱
    if context is not None:
        nc = context.run_config.get("node-config") or context.run_config.get("node_config") or context.run_config.get("node_config_str")
        if nc:
            parsed = parse_node_config_string(nc if isinstance(nc, str) else str(nc))
            pid = parsed.get("partition-id") or parsed.get("partition_id")
            nparts = parsed.get("num-partitions") or parsed.get("num_partitions")
            if pid is not None:
                pid = int(pid)
                nparts = int(nparts) if nparts is not None else 1
                return pid, nparts

    # 3) 환경변수 폴백 (선택적)
    nc_env = os.environ.get("NODE_CONFIG")
    if nc_env:
        parsed = parse_node_config_string(nc_env)
        pid = parsed.get("partition-id") or parsed.get("partition_id")
        nparts = parsed.get("num-partitions") or parsed.get("num_partitions")
        if pid is not None:
            return int(pid), int(nparts) if nparts is not None else 1

    # 4) 최종 폴백: partition 0, 1개 파티션
    return 0, 1

def cleanup_power_monitor(power_monitor, interface, device_name, start_net):
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Stopping power monitor via atexit...")
    if power_monitor:
        elapsed_time, data_size = power_monitor.stop()
        if elapsed_time is not None:
            logger.info(f"Measured power consumption: Duration={elapsed_time}s, Data size={data_size} samples.")
            power_monitor.save(_LOCAL_PATH+f"power_{device_name}_{datetime.now().strftime('%Y-%m-%d_%H%M%S.%f')}.csv")
            power_monitor.close()
        else:
            logger.warning("Power monitoring failed or returned no data.")
    end_time = time.time()
    logger.info([f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Communication end: {end_time}"])
    
def log_phase_event(tag: str, sent: int = 0, recv: int = 0):
    """CSV 로그 한 줄 기록"""
    os.makedirs(_LOCAL_PATH, exist_ok=True)
    with open(_LOCAL_STATE["fl_csv_fname"], 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            _LOCAL_STATE["device_name"]+"-"+tag, sent, recv
        ])

def initialize_state(context: Context):
    """초기화: device_name, interfaces, power monitor 등 설정"""
    global _LOCAL_STATE
    model_name       = context.run_config["model"]
    batch_size       = int(cfg(context, "batch-size", 32))
    dataset_name     = context.run_config["dataset"]
    local_epochs     = int(cfg(context, "local-epochs", 1))
    grad_ckpt        = int(cfg(context, "gradient-checkpointing", False))
    quantization     = int(cfg(context, "quantization", 4))
    num_rounds       = int(cfg(context, "num-server-rounds", 1))
    lr_max           = float(cfg(context, "learning-rate-max", 5e-5))
    lr_min           = float(cfg(context, "learning-rate-min", 1e-6))
    seq_len          = int(cfg(context, "seq-len", 0)) 
    lora_cfg         = lora_cfg_from(context)
    train_arg        = training_arguments_from(context)
    try:
        partition_id, num_partitions = get_partition_id_from_ip(context)
    except ValueError as e:
        logger.warning(f"{e} → fallback to partition_id=0")
        partition_id, num_partitions = 0, 1

    # Dataset & model
    dataset, _ = load_data(dataset_name, partition_id, num_partitions, batch_size, model_name)
    tokenizer, data_collator, formatting_prompts = get_tokenizer_and_data_collator_and_propt_formatting(model_name)

    num_classes = get_num_classes(dataset_name)
    net = get_model(model_name, num_classes, dataset_name, lora_cfg, grad_ckpt, quantization)

    _LOCAL_STATE.update({
        "dataset": dataset,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "formatting_prompts": formatting_prompts,
        "net": net,
        "local_epochs": local_epochs,
        "num_rounds": num_rounds,
        "lr_min": lr_min,
        "lr_max": lr_max,
        "lora_cfg": lora_cfg,
        "train_arg": train_arg,
        "seq_len": seq_len,
    })
    # Initialize net usage
    try:
        _LOCAL_STATE["net_start"] = get_network_usage(_LOCAL_STATE["interfaces"])
    except Exception:
        _LOCAL_STATE["net_start"] = {"bytes_sent": 0, "bytes_recv": 0}

    _LOCAL_STATE["initialized"] = True
    logger.info("Initialization completed")

def phase_start(phase_name: str):
    """phase 시작: 전력/네트워크 기록 초기화"""
    logger.info(f"[Phase] {phase_name}: starting...")
    global _LOCAL_STATE
    _LOCAL_STATE[f"{phase_name}_t0"] = time.time()
    pm = _LOCAL_STATE.get("power_monitor")
    if pm:
        logger.info(f"[PM] {phase_name}: starting power monitor...")
        pm.start(freq=0.005) # 5ms 간격, 200MHz
        time.sleep(0.5)

    try:
        _LOCAL_STATE["net_start"] = get_network_usage(_LOCAL_STATE["interfaces"])
        logger.info(f"[Network] {phase_name}: starting network usage monitoring...")
    except Exception as e:
        logger.warning(f"Failed to get network usage start for {phase_name}: {e}")
        _LOCAL_STATE["net_start"] = {"bytes_sent": 0, "bytes_recv": 0}

    #log_phase_event(f"{phase_name}_start")
    return _LOCAL_STATE[f"{phase_name}_t0"]


def phase_end(phase_name: str):
    """phase 종료: 전력/네트워크 기록 저장"""
    t1 = time.time()
    logger.info(f"[Phase] {phase_name}: ending...")
    global _LOCAL_STATE
    if f"{phase_name}_t0" in _LOCAL_STATE:
        t0 = _LOCAL_STATE[f"{phase_name}_t0"]
    else:
        t0 = 0
    duration = t1 - t0
    pm = _LOCAL_STATE.get("power_monitor")
    sent = recv = 0
    power = 0

    # Power stop
    if pm:
        elapsed_time, data_size = pm.stop()
        if elapsed_time:
            logger.info(f"[PM] {phase_name}: Duration={elapsed_time:.2f}s, Samples={data_size}")
            pm.save(_LOCAL_PATH+f"{phase_name}_power_{_LOCAL_STATE['device_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            power = pm.read_power_avg()
        else:
            logger.warning(f"[PM] {phase_name}: Power monitoring from {_LOCAL_STATE['power']} returned no data.")

    # Network usage
    try:
        net_end = get_network_usage(_LOCAL_STATE["interfaces"])
        net_start = _LOCAL_STATE.get("net_start", {"bytes_sent": 0, "bytes_recv": 0})
        sent = net_end["bytes_sent"] - net_start["bytes_sent"]
        recv = net_end["bytes_recv"] - net_start["bytes_recv"]
        _LOCAL_STATE["net_start"] = net_end
        logger.info(f"[Network] {phase_name}: Sent={sent} bytes, Received={recv} bytes")
    except Exception as e:
        logger.warning(f"Failed to get network usage end for {phase_name}: {e}")

    #log_phase_event(f"{phase_name}_end", sent, recv)
    return sent, recv, power, duration

##############################################################################################################

# Flower ClientApp
app = ClientApp()

##############################################################################################################

@app.lifespan()
def lifespan(context: Context):
    logger.info("[lifespan] Starting client...")

    pid = psutil.Process().ppid()
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] PPID: {pid}")
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # 안전하게 last octet 얻기 (실패 시 '0' 사용)
    try:
        last_oct = get_last_octet_from_ip()
    except Exception:
        last_oct = 0
    _, device_name_placeholder, _ = get_device_name()
    # 로그 파일명에 placeholder 사용 (실제 device_name은 train/evaluate에서 덮어씌움)
    logging.basicConfig(filename=_LOCAL_PATH+f"fl_info_{current_time}_{device_name_placeholder}_{last_oct}.txt")
    try:
        fl.common.logger.configure(identifier="myFlowerExperiment", filename=_LOCAL_PATH+f"fl_log_{current_time}_{device_name_placeholder}_{last_oct}.txt")
    except Exception:
        logger.warning("fl.common.logger.configure failed (continuing without fl log file)")
    logger.info([f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Client Start!"])

    # 로컬 임시 상태 저장소 (RecordDict 제약을 피하기 위해 사용)
    global _LOCAL_STATE
    _LOCAL_STATE = {}

    interfaces, device_name, power_info = get_device_name()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # CSV 로그 파일명
    fl_csv_fname = f"fl_{datetime.now().strftime('%Y%m%d')}_{device_name}_{get_last_octet_from_ip()}.csv"

    _LOCAL_STATE.update({
        "interfaces": interfaces,
        "device_name": device_name,
        "power": power_info,
        "fl_csv_fname": fl_csv_fname,
        "device": device,
    })
        # Optional: Power monitor
    if context.run_config.get("power-monitor", False) and "power_monitor" not in _LOCAL_STATE:
        logger.info(f"[PM]: initializing power monitor...")
        _LOCAL_STATE["power_monitor"] = get_power_monitor(power_info, device_name=socket.gethostname())
    else:
        _LOCAL_STATE["power_monitor"] = None
    
    yield

    cleanup_power_monitor(
        _LOCAL_STATE.get("power_monitor"), _LOCAL_STATE.get("interfaces"),
        _LOCAL_STATE.get("device_name"), _LOCAL_STATE.get("net_start")
    )

@app.train()
def train(msg: Message, context: Context) -> Message:
    global _LOCAL_STATE
    md = msg.metadata
    try:
        server_round = int(md.group_id)  # 만약 그룹ID가 라운드 문자열이라면
    except Exception:
        # fallback: 직접 content 안에 있을 수 있으니
        server_round = msg.content.get("server-round", 1) if hasattr(msg.content, "get") else 1

    logger.info(f"[TRA] server_round = {server_round}")
    if server_round == 1:
        '''
        1. Initialization Phase (First round)
        '''
        initialize_state(context)
        update_sent, update_recv, update_power, update_end_time = 0, 0, 0, 0
    else:
        '''
        2. Receive Phase (After 2nd round)
        '''
        update_sent, update_recv, update_power, update_end_time = phase_end("update_end")
    
    '''
    3. Local Training Phase
    '''
    logger.info("Starting local training...")
    _ = phase_start("train_start")
    set_peft_model_state_dict(_LOCAL_STATE["net"], msg.content["arrays"].to_torch_state_dict())

    cfg_msg = msg.content.get("config", {})
    round_id = cfg_msg.get("server-round", 0)
    new_lr = cosine_annealing(
        round_id, _LOCAL_STATE["num_rounds"],
        _LOCAL_STATE["lr_max"], _LOCAL_STATE["lr_min"],
    )
    
    _LOCAL_STATE["train_arg"].learning_rate = new_lr
    _LOCAL_STATE["train_arg"].output_dir = msg.content["config"]["save_path"]

    # Construct trainer
    trainer = SFTTrainer(
        model=_LOCAL_STATE["net"],
        tokenizer=_LOCAL_STATE["tokenizer"],
        args=_LOCAL_STATE["train_arg"],
        max_seq_length=_LOCAL_STATE["seq_len"],
        train_dataset=_LOCAL_STATE["dataset"],
        formatting_func=_LOCAL_STATE["formatting_prompts"],
        data_collator=_LOCAL_STATE["data_collator"],
    )

    # Do local training
    results = trainer.train()
    train_sent, train_recv, train_power, train_duration = phase_end("train_end")
    logger.info(f"Local training completed. (Loss: {results.training_loss}, Duration: {train_duration}s)")

    upload_start_time = phase_start("upload_start")
    metrics = {
        "num-examples": len(_LOCAL_STATE["dataset"]),
        "train_loss": results.training_loss,
        "train_energy": train_power,
        "train_sent": train_sent,
        "train_recv": train_recv,
        "train_time": train_duration,
        "update_energy": update_power,
        "update_sent": update_sent,
        "update_recv": update_recv,
        "update_end_time": update_end_time,
        "upload_start_time": upload_start_time,
    }
    model_record = ArrayRecord(get_peft_model_state_dict(_LOCAL_STATE["net"]))
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
