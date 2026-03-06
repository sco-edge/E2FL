"""E2FL: A Flower / PyTorch app."""
from log import WrlsEnv
from power import _power_monitor_interface
from power.powermon import get_power_monitor

from datetime import datetime
import psutil, logging, time, socket, csv, importlib, logging, os, subprocess

import torch
import flwr as fl
from flwr.clientapp import ClientApp
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from e2fl.task_llm import set_weights, test, train as train_model, get_flops, is_lora_model, get_peft_model_state_dict
from e2fl.models import get_model, is_llm_model_name
from e2fl.dataset import load_data, get_num_classes
from e2fl.utils import get_device_name, get_last_octet_from_ip, parse_node_config_string, get_network_usage
     
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

def ensure_time_sync(
    ntp_server: str | None = None,
    max_offset_ms: float = 50.0,
):
    """
    간단한 NTP 오프셋 체크 + 필요하면 ntpdate 시도.
    - ntplib 없으면 그냥 워닝만 찍고 스킵.
    - root 권한 아니면 ntpdate는 안 돌리고 워닝만.
    """
    global logger

    ntp_server = ntp_server or os.environ.get("NTP_SERVER", "time.google.com")

    try:
        import ntplib
    except ImportError:
        logger.warning("[TimeSync] ntplib not installed; skipping NTP check")
        return

    try:
        client = ntplib.NTPClient()
        res = client.request(ntp_server, version=3, timeout=3)
        offset_ms = res.offset * 1000.0
        logger.info(f"[TimeSync] NTP offset to {ntp_server}: {offset_ms:.2f} ms")

        if abs(offset_ms) > max_offset_ms:
            logger.warning(
                f"[TimeSync] Large offset ({offset_ms:.2f} ms) detected."
            )
            # root이면 한 번 ntpdate 시도
            try:
                if hasattr(os, "geteuid") and os.geteuid() == 0:
                    logger.info(
                        f"[TimeSync] Trying to correct time via `ntpdate -u {ntp_server}`"
                    )
                    subprocess.run(
                        ["ntpdate", "-u", ntp_server],
                        check=False,
                    )
                else:
                    logger.warning(
                        "[TimeSync] Not running as root; "
                        "please configure system NTP (chrony/systemd-timesyncd) manually."
                    )
            except Exception as e2:
                logger.warning(f"[TimeSync] Failed to run ntpdate: {e2}")
    except Exception as e:
        logger.warning(f"[TimeSync] NTP check/sync failed: {e}")

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
    end_time = time.perf_counter()
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
    model_name = context.run_config["model"]
    # Prefer 'batch_size' (server-side key), but fall back to 'batch-size' for backward compatibility
    batch_size = context.run_config.get("batch_size", context.run_config.get("batch-size", 32))
    dataset_name = context.run_config["dataset"]

    # LLMs (e.g., Gemma3 270M) do not require num_classes; use dummy value 0.
    # For vision/classification models, fall back to the original get_num_classes.
    if is_llm_model_name(model_name):
        num_classes = 0
        logger.info(f"[Init] LLM model detected ({model_name}); using num_classes=0")
    else:
        num_classes = get_num_classes(dataset_name)
    try:
        partition_id, num_partitions = get_partition_id_from_ip(context)
    except ValueError as e:
        logger.warning(f"{e} → fallback to partition_id=0")
        partition_id, num_partitions = 0, 1

    # Dataset & model
    trainloader, valloader = load_data(dataset_name, partition_id, num_partitions, batch_size, model_name)
    net, default_local_epochs = get_model(model_name, num_classes), context.run_config.get("local_epochs", 1)

    # Respect any pre-set local_epochs in _LOCAL_STATE (e.g., from per-round overrides)
    local_epochs = _LOCAL_STATE.get("local_epochs", default_local_epochs)
    logger.info(f"[Init] initialize_state using local_epochs={local_epochs}")

    _LOCAL_STATE.update({
        "trainloader": trainloader,
        "valloader": valloader,
        "net": net,
        "local_epochs": local_epochs,
        "model_name": model_name,
        "batch_size": batch_size,
        "dataset_name": dataset_name,
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
    upload_start_time = time.time()
    _LOCAL_STATE[f"{phase_name}_t0"] = time.perf_counter()
    pm = _LOCAL_STATE.get("power_monitor")
    if pm:
        logger.info(f"[PM] {phase_name}: power monitor already running (no restart).")
    else:
        logger.warning(f"[PM] {phase_name}: power monitor is not initialized.")

    try:
        _LOCAL_STATE["net_start"] = get_network_usage(_LOCAL_STATE["interfaces"])
        logger.info(f"[Network] {phase_name}: starting network usage monitoring: {_LOCAL_STATE['net_start']['bytes_sent']} bytes sent, {_LOCAL_STATE['net_start']['bytes_recv']} bytes recv")
    except Exception as e:
        logger.warning(f"Failed to get network usage start for {phase_name}: {e}")
        _LOCAL_STATE["net_start"] = {"bytes_sent": 0, "bytes_recv": 0}

    #log_phase_event(f"{phase_name}_start")
    return _LOCAL_STATE[f"{phase_name}_t0"], upload_start_time


def phase_end(phase_name: str):
    """phase 종료: 전력/네트워크 기록 저장"""
    t1 = time.perf_counter()
    logger.info(f"[Phase] {phase_name}: ending...")
    global _LOCAL_STATE
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
        logger.info(f"[Network] {phase_name}: Sent={sent} bytes, Received={recv} bytes (net_start: [bytes_sent: {net_start['bytes_sent']}, bytes_recv: {net_start['bytes_recv']}])")
    except Exception as e:
        logger.warning(f"Failed to get network usage end for {phase_name}: {e}")

    #log_phase_event(f"{phase_name}_end", sent, recv)
    return sent, recv, power, t1

##############################################################################################################
# Flower ClientApp
app = ClientApp()
##############################################################################################################

@app.lifespan()
def lifespan(context: Context):
    logger.info("[lifespan] Starting client...")

    ensure_time_sync()

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

    # Device-specific LoRA rank override (from pyproject / Flower run_config)
    # Priority:
    #   1) lora-r-<device_name> (e.g., lora-r-RPi5, lora-r-jetson_orin_nx)
    #   2) global lora-r
    # If neither is set or parsing fails, we leave the model-side default unchanged.
    try:
        run_cfg = context.run_config if hasattr(context, "run_config") else {}
        base_r_raw = run_cfg.get("lora-r", None) if isinstance(run_cfg, dict) else None
        base_r = int(base_r_raw) if base_r_raw not in (None, "") else None

        override_key = f"lora-r-{device_name}"
        dev_r_raw = run_cfg.get(override_key, None) if isinstance(run_cfg, dict) else None
        if dev_r_raw not in (None, ""):
            lora_r = int(dev_r_raw)
        else:
            lora_r = base_r

        if lora_r is not None:
            os.environ["LORA_R"] = str(lora_r)
            logger.info(f"[Init] Using LoRA rank={lora_r} for device '{device_name}'")
            _LOCAL_STATE["lora_rank"] = lora_r   # <<< 이 줄 추가
        else:
            logger.info("[Init] No explicit LoRA rank override found for this device; using model default.")
    except Exception as e:
        logger.warning(f"[Init] Failed to derive device-specific LoRA rank: {e}")

    if device_name == "jetson_orin_nx":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
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

    # Start power monitor ONCE at client startup (continuous sampling)
    pm = _LOCAL_STATE["power_monitor"]
    if pm:
        try:
            logger.info("[PM] Starting power monitor once at client startup (continuous sampling)...")
            pm.start(freq=0.005)  # 5ms sampling
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"[PM] Failed to start power monitor at startup: {e}")

    yield

    cleanup_power_monitor(
        _LOCAL_STATE.get("power_monitor"), _LOCAL_STATE.get("interfaces"),
        _LOCAL_STATE.get("device_name"), _LOCAL_STATE.get("net_start")
    )

@app.train()
def train(msg: Message, context: Context) -> Message:
    global _LOCAL_STATE
    md = msg.metadata

    # --- timing breakdown for train phase ---
    init_time = 0.0
    loop_time = 0.0

    # Extract per-round config sent from the server (ConfigRecord behaves like a mapping)
    cfg = msg.content.get("config", {})
    if hasattr(cfg, "get"):
        get_cfg = cfg.get
    else:
        # Fallback if config is missing or is not mapping-like
        def get_cfg(key, default=None):
            return default

    # Round index (used for phase handling)
    try:
        server_round = int(get_cfg("server-round", 1))
    except Exception:
        server_round = 1

    logger.info(f"[TRA] server_round = {server_round}")

    # Optional workload knobs sent by the server strategy
    try:
        # 공통 fallback
        max_samples_global = int(get_cfg("max_samples_per_round", 0) or 0)
    except Exception:
        max_samples_global = 0

    # 디바이스별 override (예: RPi5 / jetson_orin_nx)
    dev_name = _LOCAL_STATE["device_name"]
    if dev_name == "RPi5":
        key = "max_samples_per_round_RPi5"
    elif dev_name == "jetson_orin_nx":
        key = "max_samples_per_round_jetson_orin_nx"
    else:
        key = None

    if key is not None:
        try:
            max_samples_dev = int(get_cfg(key, "") or 0)
        except Exception:
            max_samples_dev = 0
    else:
        max_samples_dev = 0

    # 최종 max_samples 선택: 디바이스용이 우선, 없으면 글로벌
    max_samples_per_round = max_samples_dev or max_samples_global

    # Optional override of local epochs (supports per-device keys)
    local_epochs_override = None
    try:
        # Device-specific key has priority if present
        if dev_name == "RPi5":
            ep_key = "local_epochs_RPi5"
        elif dev_name == "jetson_orin_nx":
            ep_key = "local_epochs_jetson_orin_nx"
        else:
            ep_key = None

        if ep_key is not None:
            val = get_cfg(ep_key, None)
            if val not in (None, ""):
                local_epochs_override = int(val)

        # Fallback to global local_epochs
        if local_epochs_override is None:
            val = get_cfg("local_epochs", None)
            if val not in (None, ""):
                local_epochs_override = int(val)

        if local_epochs_override is not None:
            _LOCAL_STATE["local_epochs"] = local_epochs_override
            logger.info(
                f"[TRA] Overriding local_epochs from config ({dev_name}): "
                f"{_LOCAL_STATE['local_epochs']}"
            )
    except Exception as e:
        logger.warning(f"[TRA] Failed to parse local_epochs from config: {e}")

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
    train_t0, _ = phase_start("train_start")

    # init 여부 확인
    initialized_before = bool(_LOCAL_STATE.get("initialized"))

    # --- init timing (처음 한 번만 의미 있음) ---
    t_init0 = time.perf_counter()
    if not initialized_before:
        initialize_state(context)
    t_init1 = time.perf_counter()

    if not initialized_before:
        init_time = t_init1 - t_init0
        logger.info(f"[train] init_time={init_time:.6f}s")

    # Debug: confirm the effective local_epochs after possible override and initialization
    logger.info(f"[train] _LOCAL_STATE['local_epochs'] after init/override = {_LOCAL_STATE.get('local_epochs')}")
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
        raise RuntimeError("Train message missing 'arrays' field")

    # Ensure local state (net, dataloaders, etc.) is initialized before using it
    if not _LOCAL_STATE.get("initialized"):
        initialize_state(context)

    net = _LOCAL_STATE["net"]

    # Build effective trainloader, possibly truncated based on server-provided workload
    base_trainloader = _LOCAL_STATE["trainloader"]
    trainloader = base_trainloader
    if "max_samples_per_round" in locals() and max_samples_per_round > 0:
        try:
            from torch.utils.data import DataLoader, Subset
            dataset = base_trainloader.dataset
            num = min(max_samples_per_round, len(dataset))
            indices = list(range(num))
            subset = Subset(dataset, indices)
            trainloader = DataLoader(
                subset,
                batch_size=base_trainloader.batch_size,
                shuffle=True,
                num_workers=getattr(base_trainloader, "num_workers", 0),
                collate_fn=getattr(base_trainloader, "collate_fn", None),
            )
            logger.info(
                f"[train] Using subset of dataset: {num}/{len(dataset)} examples "
                f"(max-samples-per-round={max_samples_per_round})"
            )
        except Exception as e:
            logger.warning(
                f"[train] Failed to create subset DataLoader; falling back to full trainloader: {e}"
            )
            trainloader = base_trainloader

    # Determine how many examples are actually used for this round
    try:
        num_examples_used = len(trainloader.dataset)
    except Exception:
        # Fallback: use full base dataset length if subset length is unavailable
        num_examples_used = len(base_trainloader.dataset)

    # ------------------------------------------------------------------
    # LoRA-aware extraction of arrays from ArrayRecord
    # ------------------------------------------------------------------
    arrays = None
    if is_lora_model(net):
        logger.info(f"[train] LoRA model detected; inspecting arrays_rec type={type(arrays_rec)}")

        # Case 1: ArrayRecord wraps a dict in `.arrays`
        if hasattr(arrays_rec, "arrays") and isinstance(arrays_rec.arrays, dict):
            arrays = arrays_rec.arrays
            logger.info(f"[train] Using arrays_rec.arrays as LoRA dict; num_keys={len(arrays)}")

        # Case 2: arrays_rec itself is a dict-like object
        elif isinstance(arrays_rec, dict):
            arrays = arrays_rec
            logger.info(f"[train] arrays_rec is already dict; num_keys={len(arrays)}")

    # Generic fallback: use as_numpy/arrays/value/identity
    if arrays is None:
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

    # Normalize incoming parameters for LoRA models:
    # For LoRA, we want a dict[str, array-like] keyed only by the adapter state dict keys.
    if is_lora_model(net):
        # Local adapter keys define the "ground truth" of what we expect to set
        lora_keys = list(get_peft_model_state_dict(net).keys())

        # 1) If arrays is not a dict yet, assume it's an ordered sequence matching lora_keys.
        if not isinstance(arrays, dict):
            try:
                seq = list(arrays)
                if len(seq) != len(lora_keys):
                    logger.error(
                        f"[train] LoRA parameters length mismatch: got {len(seq)} values, expected {len(lora_keys)}"
                    )
                    raise RuntimeError("LoRA parameter length mismatch")
                arrays = {k: v for k, v in zip(lora_keys, seq)}
            except Exception as e:
                logger.error(f"[train] Failed to remap LoRA parameters into dict for LoRA model: {e}")
                raise

        # 2) Filter the dict so that:
        #    - only local LoRA keys are kept
        #    - any clearly non-tensor values (e.g., str) are dropped
        clean_arrays = {}
        dropped_non_tensor = []
        missing_keys = []

        for k in lora_keys:
            if k not in arrays:
                missing_keys.append(k)
                continue
            v = arrays[k]
            if isinstance(v, str):
                dropped_non_tensor.append((k, type(v)))
                continue
            clean_arrays[k] = v

        extra_keys = set(arrays.keys()) - set(lora_keys)
        if extra_keys:
            logger.warning(
                f"[train] Received {len(extra_keys)} extra LoRA keys not present in local adapter; "
                f"they will be ignored. Sample: {list(extra_keys)[:3]}"
            )
        if missing_keys:
            logger.warning(
                f"[train] Missing {len(missing_keys)} LoRA keys from server payload. "
                f"First few missing: {missing_keys[:3]}"
            )
        if dropped_non_tensor:
            logger.warning(
                f"[train] Dropped {len(dropped_non_tensor)} non-tensor LoRA entries before set_weights. "
                f"Sample: {dropped_non_tensor[:3]}"
            )

        arrays = clean_arrays

    # arrays는 이제 numpy 배열 리스트 또는 state list 여야 함
    set_weights(net, arrays)

    # Resolve learning rate (LLM vs vision defaults)
    model_name_cfg = context.run_config.get("model", "")
    is_llm = is_llm_model_name(model_name_cfg)
    default_lr = 5e-5 if is_llm else 0.01
    lr = float(context.run_config.get("learning-rate", default_lr))

    # Use the unified train() from e2fl.task_llm, which supports both LLM and vision models
    t_loop0 = time.perf_counter()
    train_loss = train_model(
        _LOCAL_STATE["net"],
        trainloader,
        _LOCAL_STATE["local_epochs"],
        _LOCAL_STATE["device"],
        lr=lr,
    )
    t_loop1 = time.perf_counter()
    loop_time = t_loop1 - t_loop0

    train_sent, train_recv, train_power, train_end_time = phase_end("train_end")
    total_train_time = train_end_time - train_t0

    logger.info(
        f"Local training completed. (Loss: {train_loss}, "
        f"total_train_time={total_train_time:.6f}s, "
        f"init_time={init_time:.6f}s, "
        f"loop_time={loop_time:.6f}s)"
    )

    _, upload_start_time = phase_start("upload_start")

    # For LoRA-based LLMs, only send adapter weights back to the server.
    # For non-LoRA models, fall back to the full state dict as before.
    net = _LOCAL_STATE["net"]
    if is_lora_model(net):
        arrays_out = get_peft_model_state_dict(net)
        logger.info("[TRA] Detected LoRA model – sending adapter-only state dict to server.")
    else:
        arrays_out = net.state_dict()
        logger.info("[TRA] Non-LoRA model – sending full state dict to server.")

    model_record = ArrayRecord(arrays_out)

    logger.info(f"[train] Preparing metrics: lora_rank={_LOCAL_STATE.get('lora_rank')}, device_code={device_lookup[_LOCAL_STATE['device_name']]}")

    metrics = {
        "num-examples": num_examples_used,
        "train_loss": train_loss,
        "train_energy": train_power,
        "train_sent": train_sent,
        "train_recv": train_recv,
        "train_time": total_train_time,          # 기존
        "train_init_time": init_time,            # 새로 추가
        "train_loop_time": loop_time,           # 새로 추가
        "update_energy": update_power,
        "update_sent": update_sent,
        "update_recv": update_recv,
        "update_end_time": update_end_time,
        "upload_start_time": upload_start_time,
        "flops": get_flops(_LOCAL_STATE["net"], _LOCAL_STATE["device"]),
        "device_code": device_lookup[_LOCAL_STATE["device_name"]],
        "lora_rank": _LOCAL_STATE.get("lora_rank"),  # <<< 이 줄 추가
    }
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
    if server_round == 1 and not _LOCAL_STATE.get("initialized"):
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
    eval_t0, _ = phase_start("eval_start")

    # Detect LLM vs. vision model and handle evaluation accordingly
    model_name_cfg = _LOCAL_STATE.get("model_name", context.run_config.get("model", ""))
    if is_llm_model_name(model_name_cfg):
        # For LLM fine-tuning, we effectively skip evaluation.
        # Still return a dummy num_examples > 0 so that aggregated metrics
        # in FedAvgLLM/Flower do not hit division-by-zero.
        logger.info("[evaluate] LLM model detected; skipping evaluation loop and returning dummy metrics.")
        eval_loss, eval_acc = 0.0, 0.0
        num_examples = 1
    else:
        # Vision/classification models: run normal test loop
        eval_loss, eval_acc = test(
            _LOCAL_STATE.get("net"),
            _LOCAL_STATE.get("valloader"),
            _LOCAL_STATE.get("device"),
        )
        # Safely determine number of evaluation examples
        valloader = _LOCAL_STATE.get("valloader")
        if valloader is not None and hasattr(valloader, "dataset"):
            num_examples = len(valloader.dataset)
        else:
            num_examples = 0

    eval_sent, eval_recv, eval_power, evaluate_end_time = phase_end("eval_end")
    logger.info(
        f"Evaluation completed with accuracy: {eval_acc}, Loss: {eval_loss}, Duration: {evaluate_end_time - eval_t0}s"
    )

    _, update_start_time = phase_start("update_start")
    metrics = {
        "num-examples": num_examples,
        "eval_loss": eval_loss,
        "eval_accuracy": eval_acc,
        "upload_energy": upload_power,
        "upload_sent": upload_sent,
        "upload_recv": upload_recv,
        "upload_end_time": upload_end_time,
        "eval_energy": eval_power,
        "eval_sent": eval_sent,
        "eval_recv": eval_recv,
        "eval_time": evaluate_end_time - eval_t0,
        "update_start_time": update_start_time,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
        