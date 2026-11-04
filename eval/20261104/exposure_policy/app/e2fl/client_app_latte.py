"""E2FL: A Flower / PyTorch app."""
from log import WrlsEnv
from power import _power_monitor_interface
from power.powermon import get_power_monitor

from datetime import datetime
import psutil, logging, time, socket, csv, importlib, logging, os, json, subprocess, re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import flwr as fl
from flwr.common import Scalar
from flwr.clientapp import ClientApp
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from e2fl.task import set_weights, test, train, get_flops
from e2fl.models import get_model
from e2fl.dataset import load_data, get_num_classes
from e2fl.utils import get_device_name, get_last_octet_from_ip, parse_node_config_string, get_network_usage

from e2fl.LAT2E.latte_selector import LatteLayerSelector
from e2fl.LAT2E.latte_predictor import LatteFinePredictor
from e2fl.LAT2E.latte_submodel import SubModelConfig, load_submodel_table, select_best_submodel
from e2fl.LAT2E.latte_freq import init_frequency_table_from_model, update_frequency_table

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

home_dir = os.path.expanduser("~")
_LOCAL_PATH = f"{home_dir}/EEFL/E2FL/eval/"

# 전역 logger 설정 (CustomFormatter 사용)
logger = logging.getLogger("e2fl_client")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)


device_lookup = {
    "RPi5": 1,
    "jetson_orin_nx": 2,
}

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

def load_latte_predictor(config, device_name):
    """
    latte-mode = 'fine'에서만 사용됨.

    - {home}/EEFL/E2FL/predictor/profile/{device_name}/{model_name}_betas_fine.json
    - {home}/EEFL/E2FL/predictor/profile/{model_name}_workload_fine.json

    을 로드해서:
      * LatteFinePredictor(betas)
      * workload(algo_selection, C_key, C_non)

    를 반환한다.
    """
    model_name = config.get("model")
    base_dir = Path.home() / "EEFL" / "E2FL" / "predictor" / "profile"

    beta_path = base_dir / device_name / f"{model_name}_betas_fine.json"
    workload_path = base_dir / f"{model_name}_workload_fine.json"

    # 1) betas_fine.json 로드 → predictor 생성
    with open(beta_path, "r") as f:
        betas = json.load(f)
    predictor = LatteFinePredictor(betas)

    # 2) workload_fine.json 로드 → 전체 모델 workload
    with open(workload_path, "r") as f:
        workload = json.load(f)

    algo_sel = workload["algo_selection"]
    C_key = workload["C_key"]
    C_non = workload["C_non"]

    # 3) (optional) MLP selector 로드
    selector = None
    mlp_path = config.get("latte-mlp-path") or config.get("latte_mlp_path")
    if mlp_path:
        try:
            selector = LatteLayerSelector(mlp_path)
            logger.info(f"[LATTE] Loaded MLP selector from {mlp_path}")
        except Exception as e:
            logger.warning(f"[LATTE] Failed to load MLP selector from {mlp_path}: {e}")

    # predictor, selector, workload를 모두 돌려줌
    return predictor, selector, algo_sel, C_key, C_non


def initialize_state(context: Context):
    """초기화: device_name, interfaces, power monitor 등 설정"""
    global _LOCAL_STATE
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
    trainloader, valloader = load_data(dataset_name, partition_id, num_partitions, batch_size, model_name)
    net, local_epochs = get_model(model_name, num_classes), context.run_config.get("local-epochs", 1)

    _LOCAL_STATE.update({
        "trainloader": trainloader,
        "valloader": valloader,
        "net": net,
        "local_epochs": local_epochs,
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
        power = pm.read_power_avg()
        logger.info(f"[PM] {phase_name}: Average Power Consumption: {power}")
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


# --- Bandwidth estimation helper (LATTE fine mode) ---
def estimate_bandwidth_bmon(interfaces=None, default_bps: float = 10_000_000.0) -> float:
    """
    Use bmon to estimate current bandwidth (bytes/sec), similar to LATTE's design.

    We invoke:
        bmon -o ascii:quitafter=1

    Then parse the ASCII output. If a specific interface name is provided, we try
    to pick that line; otherwise, we use the first interface we can parse.

    If bmon is not available or parsing fails, we fall back to default_bps.
    """
    # Decide which interface name to look for
    iface_name = None
    if isinstance(interfaces, (list, tuple)):
        if interfaces:
            iface_name = interfaces[0]
    elif isinstance(interfaces, str):
        iface_name = interfaces

    cmd = ["bmon", "-o", "ascii:quitafter=1"]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3.0,
        )
    except FileNotFoundError:
        logger.warning("[LATTE] bmon not found in PATH, using default bandwidth")
        return float(default_bps)
    except Exception as e:
        logger.warning(f"[LATTE] bmon invocation failed ({e}), using default bandwidth")
        return float(default_bps)

    output = proc.stdout.splitlines()
    if not output:
        logger.warning("[LATTE] bmon produced no output, using default bandwidth")
        return float(default_bps)

    # Simple heuristic parser:
    # Look for a line containing the interface name (if given),
    # then parse numbers that look like '123.4 kbit/s', '10.2 Mbit/s', etc.
    target_line = None
    for line in output:
        if iface_name is not None:
            if iface_name in line:
                target_line = line
                break
        else:
            # If no specific iface, pick the first non-empty, non-header line
            if line.strip():
                target_line = line
                break

    if target_line is None:
        logger.warning("[LATTE] Could not find interface line in bmon output, using default bandwidth")
        return float(default_bps)

    # Regex to capture things like: 123.4 kbit/s or 10.2 Mbit/s
    matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*([kMGT]?bit/s)", target_line)
    if not matches:
        logger.warning("[LATTE] Failed to parse rates from bmon output line, using default bandwidth")
        return float(default_bps)

    # Convert the first match to bits/sec
    val_str, unit = matches[0]
    try:
        val = float(val_str)
    except ValueError:
        logger.warning("[LATTE] Failed to parse numeric rate from bmon output, using default bandwidth")
        return float(default_bps)

    # Unit scaling
    factor = 1.0
    if unit.startswith("k"):
        factor = 1e3
    elif unit.startswith("M"):
        factor = 1e6
    elif unit.startswith("G"):
        factor = 1e9
    elif unit.startswith("T"):
        factor = 1e12

    bits_per_sec = val * factor
    bytes_per_sec = bits_per_sec / 8.0
    logger.info(f"[LATTE] Estimated bandwidth via bmon: {bytes_per_sec:.2f} bytes/sec (from '{target_line.strip()}')")
    return float(bytes_per_sec)

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
    
    if context.run_config.get("latte-mode", "coarse") == "fine":
        try:
            predictor, selector, algo_sel, C_key, C_non = load_latte_predictor(
                context.run_config, device_name
            )
            _LOCAL_STATE["latte_predictor"] = predictor
            _LOCAL_STATE["latte_algo_selection"] = algo_sel
            _LOCAL_STATE["latte_C_key_full"] = C_key
            _LOCAL_STATE["latte_C_non_full"] = C_non

            if selector is not None:
                _LOCAL_STATE["latte_selector"] = selector
                logger.info("[LATTE] MLP layer selector initialized and stored in _LOCAL_STATE['latte_selector']")
            else:
                logger.info("[LATTE] No MLP selector configured (latte-mlp-path not set)")

        except FileNotFoundError as e:
            logger.warning(f"[LATTE] Fine-mode JSON not found: {e} → falling back to coarse mode")
        except Exception as e:
            logger.warning(f"[LATTE] Failed to initialize fine-mode predictor: {e}")

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
        server_round = int(msg.content["config"]["server-round"])
    except Exception:
        # fallback: 직접 content 안에 있을 수 있으니
        server_round = msg.content.get("server-round", 1) if hasattr(msg.content, "get") else 1

    logger.info(f"[TRA] server_round = {server_round}")
    # Per-round config from server (for LATTE fine mode)
    cfg = msg.content.get("config", {}) if msg.has_content() else {}
    latte_mode = context.run_config.get("latte-mode", "coarse")
    if server_round == 1:
        '''
        1. Initialization Phase (First round)
        '''
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

    try:
        arrays_rec = msg.content["arrays"]
        logger.info("[DEBUG] arrays_rec fetched from msg.content['arrays']")
    except Exception as e:
        logger.info(f"[DEBUG] FAILED: msg.content['arrays']: {e}")
        arrays_rec = None

    # fallback checks
    if arrays_rec is None:
        try:
            arrays_rec = getattr(msg, "arrays")
            logger.info("[DEBUG] arrays_rec fetched from msg.arrays attribute")
        except Exception as e:
            logger.info(f"[DEBUG] FAILED: getattr(msg,'arrays'): {e}")
            arrays_rec = None

    # If still None → definitely the crash source
    if arrays_rec is None:
        logger.error("[FATAL] Could not retrieve arrays from message. Train cannot continue.")
        # You can raise an exception or return a dummy Message to avoid crash
        raise RuntimeError("Train message missing 'arrays' field")

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
            
    initialize_state(context)
    latte_mode = context.run_config.get("latte-mode", "coarse")
    # --- LATTE fine-mode submodel and frequency logic ---
    if latte_mode == "fine":
        # 1) Load S table from cfg into _LOCAL_STATE["latte_S"] if not already
        if "latte_S" not in _LOCAL_STATE and isinstance(cfg, dict) and "latte_S" in cfg:
            try:
                raw_S = cfg["latte_S"]
                _LOCAL_STATE["latte_S"] = [SubModelConfig.from_dict(d) for d in raw_S]
                logger.info(f"[LATTE] Loaded submodel table S with {len(_LOCAL_STATE['latte_S'])} entries")
            except Exception as e:
                logger.warning(f"[LATTE] Failed to deserialize latte_S from config: {e}")

        # 2) Initialize frequency table
        if "latte_F" not in _LOCAL_STATE:
            try:
                F0 = init_frequency_table_from_model(_LOCAL_STATE["net"])
                _LOCAL_STATE["latte_F"] = F0
                logger.info("[LATTE] Initialized frequency table F from model")
            except Exception as e:
                logger.warning(f"[LATTE] Failed to initialize frequency table F: {e}")

        # 3) Submodel scheduling
        S = _LOCAL_STATE.get("latte_S")
        F = _LOCAL_STATE.get("latte_F")
        predictor = _LOCAL_STATE.get("latte_predictor")

        pred_time = None  # default

        if S and predictor is not None:
            try:
                U = float(cfg.get("latte-deadline", 5.0))
            except Exception:
                U = 5.0

            # 현재 대역폭 측정
            B = estimate_bandwidth_bmon(_LOCAL_STATE.get("interfaces"))

            num_epochs = int(_LOCAL_STATE.get("local_epochs", 1))
            batch_size = int(context.run_config.get("batch-size", 32))
            try:
                num_batches = len(_LOCAL_STATE["trainloader"])
            except Exception:
                num_batches = 1

            # 전체 모델 파라미터 크기(바이트) 추정
            try:
                base_param_size_bytes = sum(
                    p.numel() * p.element_size() for p in _LOCAL_STATE["net"].parameters()
                )
            except Exception:
                base_param_size_bytes = 0

            try:
                best_cfg, best_pred = select_best_submodel(
                    submodels=_LOCAL_STATE["latte_S"],
                    predictor=_LOCAL_STATE["latte_predictor"],
                    algo_selection=_LOCAL_STATE["latte_algo_selection"],
                    C_key_full=_LOCAL_STATE["latte_C_key_full"],
                    C_non_full=_LOCAL_STATE["latte_C_non_full"],
                    B_bytes_per_sec=float(B),
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    num_batches=num_batches,
                    deadline_U=U,
                    base_param_size_bytes=base_param_size_bytes,
                )
            except Exception as e:
                logger.warning(f"[LATTE] select_best_submodel failed: {e}")
                best_cfg, best_pred = None, None

            if best_cfg is not None:
                _LOCAL_STATE["latte_submodel_id"] = best_cfg.id
                # Frequency table update (윈도우 기준)
                try:
                    if F is not None:
                        update_frequency_table(F, best_cfg.windows)
                except Exception as e:
                    logger.warning(f"[LATTE] Failed to update frequency table F: {e}")

            if best_pred is not None:
                _LOCAL_STATE["latte_T_train_pred"] = best_pred.get("T_train")
                _LOCAL_STATE["latte_T_total_pred"] = best_pred.get("T_total")
                pred_time = _LOCAL_STATE["latte_T_train_pred"]
        else:
            pred_time = None


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
        "flops": get_flops(_LOCAL_STATE["net"], _LOCAL_STATE["device"]),
        "device_code": device_lookup[_LOCAL_STATE["device_name"]],
        "latte_pred_time": float(pred_time) if pred_time is not None else -1.0,
    }
    # --- LATTE fine-mode metrics ---
    if latte_mode == "fine":
        pred_time = _LOCAL_STATE.get("latte_T_train_pred")
        sub_id = _LOCAL_STATE.get("latte_submodel_id")
        if pred_time is not None:
            metrics["pred_time"] = float(pred_time)
        if sub_id is not None:
            metrics["latte_submodel_id"] = int(sub_id)
    metric_record = MetricRecord(metrics)
    config_rec = msg.content.get("config", {})
    content = RecordDict({
        "arrays": model_record, 
        "metrics": metric_record,
        "config": config_rec,
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    global _LOCAL_STATE
    md = msg.metadata
    try:
        server_round = int(md.group_id)  # 만약 그룹ID가 라운드 문자열이라면
    except Exception:
        # fallback: 직접 content 안에 있을 수 있으니
        server_round = msg.content.get("server-round", 1) if hasattr(msg.content, "get") else 1

    logger.info(f"[Eval] server_round = {server_round}")
    if server_round == 1:
        '''
        1. Initialization Phase (First round)
        '''
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
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
        