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
from e2fl.dataset import load_data, get_num_classes
from e2fl.utils import get_device_name, get_last_octet_from_ip, parse_node_config_string, get_network_usage
from peft import get_peft_model_state_dict, set_peft_model_state_dict

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
    # Log the network IO
    end_net = get_network_usage(interface)
    net_usage_sent = end_net["bytes_sent"] - start_net["bytes_sent"]
    net_usage_recv = end_net["bytes_recv"] - start_net["bytes_recv"]
    logger.info([f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] Evaluation phase ({interface}): [sent: {net_usage_sent}, recv: {net_usage_recv}]"])

##############################################################################################################
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
_LOCAL_STATE = {}

# Flower ClientApp
app = ClientApp()

##############################################################################################################

def log_phase_event(tag: str, sent: int = 0, recv: int = 0):
    """CSV 로그 한 줄 기록"""
    with open(_LOCAL_STATE["fl_csv_fname"], 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            _LOCAL_STATE["device_name"]+"-"+tag, sent, recv
        ])

def initialize_state(context: Context):
    """초기화: device_name, interfaces, power monitor 등 설정"""
    model_name, batch_size, dataset_name = (
        context.run_config["model"],
        context.run_config.get("batch-size", 32),
        context.run_config["dataset"],
    )
    num_classes = get_num_classes(dataset_name)
    try:
        partition_id, num_partitions = get_partition_id_from_ip(context)
    except ValueError as e:
        logger.warning(f"{e} → fallback to partition_id=0")
        partition_id, num_partitions = 0, 1

    # Dataset & model
    trainloader, valloader = load_data(dataset_name, partition_id, num_partitions, batch_size)
    net, local_epochs = get_model(model_name, num_classes), context.run_config.get("local-epochs", 1)

    # Device info
    interfaces, device_name, power_info = get_device_name()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # CSV 로그 파일명
    fl_csv_fname = f"fl_{datetime.now().strftime('%Y%m%d')}_{device_name}_{get_last_octet_from_ip()}.csv"

    _LOCAL_STATE.update({
        "trainloader": trainloader,
        "valloader": valloader,
        "net": net,
        "local_epochs": local_epochs,
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
    
    # Initialize net usage
    try:
        _LOCAL_STATE["net_start"] = get_network_usage(interfaces)
    except Exception:
        _LOCAL_STATE["net_start"] = {"bytes_sent": 0, "bytes_recv": 0}

    _LOCAL_STATE["initialized"] = True
    logger.info("Initialization completed")
    #return _LOCAL_STATE

def phase_start(phase_name: str):
    """phase 시작: 전력/네트워크 기록 초기화"""
    logger.info(f"[Phase] {phase_name}: starting...")
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
@app.train()
def train(msg: Message, context: Context) -> Message:
    if "net" not in _LOCAL_STATE:
        '''
        1. Initialization Phase (First round)
        '''
        logger.info("Initializing for train()")
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
    arrays_rec = msg.content["arrays"]
    try:
        arrays = arrays_rec.as_numpy()          # older API
    except Exception:
        if hasattr(arrays_rec, "arrays"):      # flwr might expose .arrays
            arrays = arrays_rec.arrays
        elif hasattr(arrays_rec, "value"):     # alternative name
            arrays = arrays_rec.value
        else:
            # 마지막 수단: 객체 자체가 리스트/ndarray일 수 있음
            arrays = arrays_rec
    # arrays는 이제 numpy 배열 리스트 또는 state list 여야 함
    set_weights(_LOCAL_STATE["net"], arrays)
    task_mod = importlib.import_module("e2fl.task")
    train_func = getattr(task_mod, "train", None) or getattr(task_mod, "train_model", None)
    if train_func is None:
        raise RuntimeError("No train or train_model() function found in e2fl.task")
    train_loss = train_func(_LOCAL_STATE["net"], _LOCAL_STATE["trainloader"], _LOCAL_STATE["local_epochs"], _LOCAL_STATE["device"])
    train_sent, train_recv, train_power, train_duration = phase_end("train_end")
    logger.info(f"Local training completed. (Loss: {train_loss}, Duration: {train_duration}s)")

    upload_start_time = phase_start("upload_start")
    if is_lora_model(_LOCAL_STATE["net"]):
        logger.info("Detected LoRA model: sending adapter weights only")
        sd = get_peft_model_state_dict(_LOCAL_STATE["net"])
        arrays = [v.detach().cpu().to(torch.float32).numpy() for k, v in sorted(sd.items())]
        model_record = ArrayRecord(arrays)
    else:
        model_record = ArrayRecord(_LOCAL_STATE["net"].state_dict())
    metrics = {
        "num-examples": len(_LOCAL_STATE["trainloader"].dataset),
        "train_loss": train_loss,
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
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    if "net" not in _LOCAL_STATE:
        '''
        1. Initialization Phase (First round)
        '''
        logger.info("Initializing for evaluate()")
        initialize_state(context)
    '''
    2. Upload Phase (After train())
    '''
    upload_sent, upload_recv, upload_power, upload_end_time = phase_end("upload_end")

    '''
    3. Evaluation Phase
    '''
    _ = phase_start("eval_start")
    eval_loss, eval_acc = test(_LOCAL_STATE.get("net"), _LOCAL_STATE.get("valloader"), _LOCAL_STATE.get("device"))
    eval_sent, eval_recv, eval_power, eval_duration = phase_end("eval_end")
    logger.info(f"Evaluation completed with accuracy: {eval_acc}, Loss: {eval_loss}, Duration: {eval_duration}s")
    
    update_start_time = phase_start("update_start")
    if is_lora_model(_LOCAL_STATE["net"]):
        sd = get_peft_model_state_dict(_LOCAL_STATE["net"])
        arrays = [v.detach().cpu().to(torch.float32).numpy() for k, v in sorted(sd.items())]
        model_record = ArrayRecord(arrays)
    else:
        model_record = ArrayRecord(_LOCAL_STATE["net"].state_dict())
    metrics = {
        "num-examples": len(_LOCAL_STATE["valloader"].dataset),
        "eval_loss": eval_loss,
        "eval_accuracy": eval_acc,
        "upload_energy": upload_power,
        "upload_sent": upload_sent,
        "upload_recv": upload_recv,
        "upload_end_time": upload_end_time,        
        "eval_energy": eval_power,
        "eval_sent": eval_sent,
        "eval_recv": eval_recv,
        "eval_time": eval_duration,
        "update_start_time": update_start_time,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
        