"""LoRACast Flower ClientApp for federated LLM fine-tuning."""
import os as _isolated_os
from pathlib import Path as _IsolatedPath
_ISOLATED_APP_ROOT = _IsolatedPath(__file__).resolve().parents[1]
_requested_root = _isolated_os.environ.get("LORACAST_WORKSPACE_ROOT")
if _requested_root and _IsolatedPath(_requested_root).resolve() != _ISOLATED_APP_ROOT:
    raise RuntimeError("isolated Flower app refuses an external workspace root")
_isolated_os.environ["LORACAST_WORKSPACE_ROOT"] = str(_ISOLATED_APP_ROOT)
from log import WrlsEnv
from power import _power_monitor_interface
from power.powermon import get_power_monitor

from datetime import datetime
import psutil, logging, time, socket, csv, importlib, logging, os, subprocess, random, hashlib, json
from pathlib import Path

import torch
import numpy as np
import flwr as fl
from flwr.clientapp import ClientApp
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from e2fl.task_llm import set_weights, load_lora_arrayrecord, test, train as train_model, get_flops, is_lora_model, get_peft_model_state_dict
from e2fl.runtime.llm_model import get_model, is_llm_model_name
from e2fl.runtime.llm_data import load_data
from e2fl.utils import get_device_name, get_last_octet_from_ip, parse_node_config_string, get_network_usage, get_stable_hostname_id
     
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
_WORKSPACE_ROOT = os.environ.get("LORACAST_WORKSPACE_ROOT", f"{home_dir}/EEFL/E2FL")
_LOCAL_PATH = os.path.join(_WORKSPACE_ROOT, "eval") + os.sep

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
    "jetson_orin_nano_super": 3,
}

# Helper for config key suffix
def _safe_config_suffix(value: str) -> str:
    """Convert device_uid/device_name into a config-key-safe suffix."""
    return str(value).replace("-", "_").replace(".", "_").replace(":", "_")

def _probe_event(context, server_round: int, event: str, **fields) -> None:
    """Run-scoped observation; no change to model, RPC, or policy decisions."""
    run_id = str(context.run_config.get("experiment-id", ""))
    if not run_id.startswith("crossclient-cap"):
        return
    uid = _safe_config_suffix(_LOCAL_STATE.get("device_uid", "unknown"))
    output = Path(_WORKSPACE_ROOT) / "eval" / "experiments" / run_id / "clients"
    output.mkdir(parents=True, exist_ok=True)
    row = {"event": event, "round": int(server_round), "device_uid": uid,
           "monotonic_s": time.monotonic(), "unix_s": time.time(), **fields}
    with (output / f"probe_phases_{uid}.jsonl").open("a", buffering=1) as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")

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
    # Manifest experiments require the explicit Flower node identity. IP modulo
    # can assign the same partition to two devices and invalidate Non-IID runs.
    if context is not None and context.run_config.get("partition-manifest-path"):
        node_config = getattr(context, "node_config", None) or {}
        pid = node_config.get("partition-id", node_config.get("partition_id"))
        nparts = node_config.get("num-partitions", node_config.get("num_partitions"))
        if pid is None:
            nc = context.run_config.get("node-config") or context.run_config.get("node_config")
            if nc:
                parsed = parse_node_config_string(str(nc))
                pid = parsed.get("partition-id", parsed.get("partition_id"))
                nparts = parsed.get("num-partitions", parsed.get("num_partitions"))
        if pid is None:
            raise ValueError("manifest experiments require an explicit node partition-id")
        if nparts is None:
            nparts = context.run_config.get("num_partitions")
        if nparts is None:
            raise ValueError("manifest experiments require num-partitions")
        return int(pid), int(nparts)

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
    """Initialize the LoRACast LLM model and the local instruction partition."""
    global _LOCAL_STATE

    run_config = context.run_config
    model_name = str(run_config["model"])
    if not is_llm_model_name(model_name):
        raise ValueError(
            f"LoRACast registration ClientApp supports LLM models only: {model_name!r}"
        )

    batch_size = int(
        run_config.get("batch_size", run_config.get("batch-size", 1))
    )
    dataset_name = str(run_config["dataset"])
    max_seq_len = int(run_config.get("max-seq-len", 256))
    partition_manifest_path = str(run_config.get("partition-manifest-path", "") or "")
    manifest_sha256 = ""
    if partition_manifest_path:
        manifest_file = Path(partition_manifest_path)
        if not manifest_file.is_absolute():
            manifest_file = Path(_WORKSPACE_ROOT) / manifest_file
        manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

    try:
        partition_id, num_partitions = get_partition_id_from_ip(context)
    except ValueError as exc:
        if partition_manifest_path:
            raise
        logger.warning("%s -> fallback to partition_id=0", exc)
        partition_id, num_partitions = 0, 1

    if run_config.get("experiment-id"):
        seed = int(run_config.get("experiment-seed", run_config.get("sample-seed", 0))) + partition_id
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info("[Client] Experiment initialization seed: %d", seed)

    device_name = str(_LOCAL_STATE.get("device_name", ""))
    rank = run_config.get(
        f"lora-r-{device_name}",
        run_config.get("lora-r", 16),
    )
    lora_config = {
        "r": int(rank),
        "alpha": int(run_config.get("lora-alpha", 32)),
        "dropout": float(run_config.get("lora-dropout", 0.05)),
        "target_modules": str(
            run_config.get(
                "lora-target-modules",
                "q_proj,k_proj,v_proj,o_proj",
            )
        ),
    }

    quantization = int(run_config.get("quantization", 0) or 0)
    gradient_checkpointing = bool(
        int(run_config.get("gradient-checkpointing", 0) or 0)
    )

    trainloader, valloader = load_data(
        dataset_name=dataset_name,
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        model_name=model_name,
        max_seq_len=max_seq_len,
        partition_manifest_path=partition_manifest_path or None,
    )
    net = get_model(
        model_name,
        lora_cfg=lora_config,
        gradient_checkpointing=gradient_checkpointing,
        quantization=quantization,
    )

    default_local_epochs = int(
        run_config.get("local_epochs", run_config.get("local-epochs", 1))
    )
    local_epochs = int(_LOCAL_STATE.get("local_epochs", default_local_epochs))

    _LOCAL_STATE.update(
        {
            "trainloader": trainloader,
            "valloader": valloader,
            "net": net,
            "local_epochs": local_epochs,
            "model_name": model_name,
            "batch_size": batch_size,
            "dataset_name": dataset_name,
            "max_seq_len": max_seq_len,
            "partition_id": partition_id,
            "dataset_size": len(trainloader.dataset),
            "partition_manifest_sha256": manifest_sha256,
        }
    )
    try:
        _LOCAL_STATE["net_start"] = get_network_usage(_LOCAL_STATE["interfaces"])
    except Exception:
        _LOCAL_STATE["net_start"] = {"bytes_sent": 0, "bytes_recv": 0}

    _LOCAL_STATE["initialized"] = True
    logger.info(
        "[Init] LoRACast state initialized: model=%s dataset=%s "
        "partition=%s/%s batch=%s rank=%s",
        model_name,
        dataset_name,
        partition_id,
        num_partitions,
        batch_size,
        rank,
    )

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
    global _LOCAL_PATH
    logger.info("[lifespan] Starting client...")

    _LOCAL_PATH = os.path.join(_WORKSPACE_ROOT, "eval") + os.sep
    experiment_id = str(context.run_config.get("experiment-id", "") or "").strip()
    if experiment_id:
        import re
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id):
            raise ValueError("invalid experiment-id")
        output_root = context.run_config.get("experiment-output-root") or os.path.join(
            _WORKSPACE_ROOT, "eval", "experiments"
        )
        experiment_root = os.path.abspath(os.path.expanduser(str(output_root)))
        _LOCAL_PATH = os.path.join(experiment_root, experiment_id, "clients") + os.sep
        os.makedirs(_LOCAL_PATH, exist_ok=True)

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

    try:
        device_uid = get_stable_hostname_id()
        logger.info(f"[Init] Using hostname-based device_uid={device_uid} for device '{device_name}'")
    except Exception as e:
        logger.warning(f"[Init] Failed to derive hostname-based device UID: {e}")
        device_uid = "unknown"

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

    if device_name in ("jetson_orin_nx", "jetson_orin_nano_super"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    # CSV 로그 파일명
    fl_csv_fname = f"fl_{datetime.now().strftime('%Y%m%d')}_{device_uid}.csv"
    if experiment_id:
        fl_csv_fname = os.path.join(_LOCAL_PATH, fl_csv_fname)

    _LOCAL_STATE.update({
        "interfaces": interfaces,
        "device_name": device_name,
        "device_uid": device_uid,
        "power": power_info,
        "fl_csv_fname": fl_csv_fname,
        "device": device,
        "experiment_id": experiment_id,
    })
    # Optional: Power monitor
    if context.run_config.get("power-monitor", False) and "power_monitor" not in _LOCAL_STATE:
        logger.info(f"[PM]: initializing power monitor...")
        _LOCAL_STATE["power_monitor"] = get_power_monitor(power_info, device_name=device_name)
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
    _probe_event(context, server_round, "client_request_handler_enter")

    # Optional workload knobs sent by the server strategy
    try:
        # 공통 fallback
        max_samples_global = int(get_cfg("max_samples_per_round", 0) or 0)
    except Exception:
        max_samples_global = 0

    # Per-client workload override has priority over device-type override.
    # Priority:
    #   1) max_samples_per_round_uid_<device_uid_safe>
    #   2) max_samples_per_round_<device_name>
    #   3) max_samples_per_round
    dev_name = _LOCAL_STATE["device_name"]
    device_uid = _LOCAL_STATE.get("device_uid", "unknown")
    uid_suffix = _safe_config_suffix(device_uid)
    uid_key = f"max_samples_per_round_uid_{uid_suffix}"

    try:
        max_samples_uid = int(get_cfg(uid_key, "") or 0)
    except Exception:
        max_samples_uid = 0

    if dev_name == "RPi5":
        key = "max_samples_per_round_RPi5"
    elif dev_name == "jetson_orin_nx":
        key = "max_samples_per_round_jetson_orin_nx"
    elif dev_name == "jetson_orin_nano_super":
        key = "max_samples_per_round_jetson_orin_nano_super"
    else:
        key = None

    if key is not None:
        try:
            max_samples_dev = int(get_cfg(key, "") or 0)
        except Exception:
            max_samples_dev = 0
    else:
        max_samples_dev = 0

    # 최종 max_samples 선택: uid용이 우선, 없으면 디바이스용, 없으면 글로벌
    max_samples_per_round = max_samples_uid or max_samples_dev or max_samples_global
    if max_samples_uid:
        logger.info(f"[TRA] Using uid-specific max_samples from config ({uid_key}): {max_samples_uid}")
    elif max_samples_dev:
        logger.info(f"[TRA] Using device-specific max_samples from config ({key}): {max_samples_dev}")
    elif max_samples_global:
        logger.info(f"[TRA] Using global max_samples_per_round from config: {max_samples_global}")

    # Optional override of local epochs.
    # Priority:
    #   1) local_epochs_uid_<device_uid_safe>
    #   2) local_epochs_<device_name>
    #   3) local_epochs
    local_epochs_override = None
    local_epochs_source = ""
    try:
        uid_ep_key = f"local_epochs_uid_{uid_suffix}"
        val = get_cfg(uid_ep_key, None)
        if val not in (None, ""):
            local_epochs_override = int(val)
            local_epochs_source = uid_ep_key

        if local_epochs_override is None:
            if dev_name == "RPi5":
                ep_key = "local_epochs_RPi5"
            elif dev_name == "jetson_orin_nx":
                ep_key = "local_epochs_jetson_orin_nx"
            elif dev_name == "jetson_orin_nano_super":
                ep_key = "local_epochs_jetson_orin_nano_super"
            else:
                ep_key = None

            if ep_key is not None:
                val = get_cfg(ep_key, None)
                if val not in (None, ""):
                    local_epochs_override = int(val)
                    local_epochs_source = ep_key

        # Fallback to global local_epochs
        if local_epochs_override is None:
            val = get_cfg("local_epochs", None)
            if val not in (None, ""):
                local_epochs_override = int(val)
                local_epochs_source = "local_epochs"

        if local_epochs_override is not None:
            _LOCAL_STATE["local_epochs"] = local_epochs_override
            logger.info(
                f"[TRA] Overriding local_epochs from config ({local_epochs_source}, {dev_name}/{device_uid}): "
                f"{_LOCAL_STATE['local_epochs']}"
            )
    except Exception as e:
        logger.warning(f"[TRA] Failed to parse local_epochs from config: {e}")

    # Optional override of local optimizer steps H.
    # Priority:
    #   1) local_steps_uid_<device_uid_safe>
    #   2) local_steps_<device_name>
    #   3) local_steps
    # A non-positive value disables the explicit step cap.
    local_steps_override = 0
    local_steps_source = "disabled"

    try:
        uid_step_key = f"local_steps_uid_{uid_suffix}"
        uid_step_value = get_cfg(uid_step_key, None)

        if uid_step_value not in (None, ""):
            local_steps_override = max(0, int(uid_step_value),)
            local_steps_source = uid_step_key

        else:
            if dev_name == "RPi5":
                device_step_key = "local_steps_RPi5"

            elif dev_name == "jetson_orin_nx":
                device_step_key = ("local_steps_jetson_orin_nx")

            elif dev_name == "jetson_orin_nano_super":
                device_step_key = ("local_steps_jetson_orin_nano_super")

            else:
                device_step_key = None

            device_step_value = (get_cfg(device_step_key, None) if device_step_key is not None else None)

            if device_step_value not in (None, ""):
                local_steps_override = max(0, int(device_step_value),)
                local_steps_source = str(device_step_key)

            else:
                global_step_value = get_cfg("local_steps", None,)

                if global_step_value not in (None, ""):
                    local_steps_override = max(0, int(global_step_value),)
                    local_steps_source = "local_steps"

        _LOCAL_STATE["local_steps"] = int(local_steps_override)

        logger.info(
            f"[TRA] Effective local_steps="
            f"{local_steps_override} "
            f"from {local_steps_source} "
            f"for {dev_name}/{device_uid}"
        )
    except Exception as e:
        _LOCAL_STATE["local_steps"] = 0

        logger.warning(
            f"[TRA] Failed to parse local_steps "
            f"from config: {e}"
        )
    
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
    _probe_event(context, server_round, "client_init_start")
    if not initialized_before:
        initialize_state(context)
    t_init1 = time.perf_counter()
    _probe_event(context, server_round, "client_init_end")

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
    exposure_tracking_enabled = bool(get_cfg("exposure_tracking_enabled", False))
    visited_sample_ids = []
    if "max_samples_per_round" in locals() and max_samples_per_round > 0:
        try:
            from torch.utils.data import DataLoader, Subset
            dataset = base_trainloader.dataset
            num = min(max_samples_per_round, len(dataset))
            indices = list(range(num))
            selection_mode = str(context.run_config.get("sample-selection-mode", "prefix"))
            shuffle_generator = None
            if selection_mode in ("round_shuffle", "probe_fixed_prefix", "exposure_aware"):
                selection_seed = (
                    int(context.run_config.get("sample-seed", 0))
                    + 1000003 * int(server_round)
                    + int(_LOCAL_STATE.get("partition_id", 0))
                )
                indices = list(range(len(dataset)))
                random.Random(selection_seed).shuffle(indices)
                if selection_mode == "exposure_aware":
                    if not exposure_tracking_enabled:
                        raise ValueError("exposure_aware sampler requires actual visit tracking")
                    exposure_dir = Path(_LOCAL_PATH) / "exposure_state" / _safe_config_suffix(device_uid)
                    exposure_dir.mkdir(parents=True, exist_ok=True)
                    visit_counts = {}
                    for prior_file in sorted(exposure_dir.glob("round_*.json")):
                        prior = json.loads(prior_file.read_text())
                        if int(prior["round"]) >= server_round:
                            continue  # a prior uncommitted attempt is never history
                        if int(prior["eligible_unique_samples"]) != len(dataset):
                            raise ValueError("dataset size changed; stable sample-ID arrival wiring not validated")
                        for sample_id in prior["visited_sample_ids"]:
                            key = int(sample_id)
                            visit_counts[key] = visit_counts.get(key, 0) + 1
                    indices.sort(key=lambda sample_id: visit_counts.get(sample_id, 0))
                indices = indices[:num]
                if selection_mode in ("round_shuffle", "exposure_aware"):
                    shuffle_generator = torch.Generator().manual_seed(selection_seed)
            elif selection_mode != "prefix":
                raise ValueError(f"unsupported sample-selection-mode: {selection_mode}")
            subset = Subset(dataset, indices)
            collate_fn = getattr(base_trainloader, "collate_fn", None)
            if exposure_tracking_enabled:
                # Record examples consumed by the actual DataLoader, including
                # partial epochs stopped by H. Local indices are stable within
                # the verified partition manifest.
                class RecordingSubset:
                    def __len__(self):
                        return len(indices)
                    def __getitem__(self, position):
                        return dict(dataset[indices[position]]), int(indices[position])
                subset = RecordingSubset()
                base_collate = collate_fn
                def collate_fn(batch):
                    rows, sample_ids = zip(*batch)
                    visited_sample_ids.extend(sample_ids)
                    return base_collate(list(rows))
            loader_kwargs = dict(
                batch_size=base_trainloader.batch_size,
                shuffle=(selection_mode != "probe_fixed_prefix"),
                num_workers=0 if exposure_tracking_enabled else getattr(base_trainloader, "num_workers", 0),
                collate_fn=collate_fn,
            )
            if shuffle_generator is not None:
                loader_kwargs["generator"] = shuffle_generator
            trainloader = DataLoader(subset, **loader_kwargs)
            if selection_mode == "probe_fixed_prefix":
                _probe_event(context, server_round, "fixed_data_prefix",
                             indices=indices, sample_count=num,
                             selection_seed=selection_seed)
            logger.info(
                f"[train] Using subset of dataset: {num}/{len(dataset)} examples "
                f"(max-samples-per-round={max_samples_per_round})"
            )
        except Exception as e:
            if context.run_config.get("partition-manifest-path") or str(
                context.run_config.get("sample-selection-mode", "prefix")
            ) != "prefix":
                raise
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

    if is_lora_model(net):
        load_lora_arrayrecord(net, arrays_rec)
        logger.info("[train] Applied all %d server LoRA tensors", len(get_peft_model_state_dict(net)))
    else:
        set_weights(net, arrays_rec.to_torch_state_dict())

    # Resolve learning rate (LLM vs vision defaults)
    model_name_cfg = context.run_config.get("model", "")
    is_llm = is_llm_model_name(model_name_cfg)
    default_lr = 5e-5 if is_llm else 0.01
    lr = float(context.run_config.get("learning-rate", default_lr))

    if context.run_config.get("experiment-id"):
        round_seed = (
            int(context.run_config.get("experiment-seed", context.run_config.get("sample-seed", 0)))
            + 1000003 * server_round + int(_LOCAL_STATE.get("partition_id", 0))
        )
        random.seed(round_seed)
        np.random.seed(round_seed % (2**32))
        torch.manual_seed(round_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(round_seed)
        logger.info("[Client] Round %d training seed: %d", server_round, round_seed)

    # Use the unified train() from e2fl.task_llm, which supports both LLM and vision models
    t_loop0 = time.perf_counter()
    _probe_event(context, server_round, "local_train_start")
    train_result = train_model(
        _LOCAL_STATE["net"],
        trainloader,
        _LOCAL_STATE["local_epochs"],
        _LOCAL_STATE["device"],
        lr=lr,
        max_steps=int(_LOCAL_STATE.get("local_steps", 0) or 0),
    )

    # Backward-compatible unpacking: older train() may return only loss
    if isinstance(train_result, tuple) and len(train_result) == 5:
        ( train_loss, input_tokens_seen, supervised_tokens_seen, executed_steps, executed_examples,) = train_result

    elif isinstance(train_result, tuple) and len(train_result) == 4:
        ( train_loss, input_tokens_seen, supervised_tokens_seen, executed_steps,) = train_result
        # Backward-compatible approximation.
        executed_examples = min(int(num_examples_used), int(executed_steps) * int(getattr(trainloader, "batch_size", 1) or 1),)
    elif isinstance(train_result, tuple) and len(train_result) == 3:
        ( train_loss, input_tokens_seen, supervised_tokens_seen, ) = train_result
        executed_steps = int(_LOCAL_STATE.get("local_steps", 0) or 0)
        executed_examples = int(num_examples_used)
    else:
        train_loss = train_result
        input_tokens_seen = 0
        supervised_tokens_seen = 0
        executed_steps = 0
        executed_examples = int(num_examples_used)
    t_loop1 = time.perf_counter()
    _probe_event(context, server_round, "local_train_end",
                 executed_steps=int(executed_steps),
                 executed_examples=int(executed_examples),
                 input_tokens=int(input_tokens_seen))
    loop_time = t_loop1 - t_loop0

    train_sent, train_recv, train_power, train_end_time = phase_end("train_end")
    total_train_time = train_end_time - train_t0

    logger.info(
        f"Local training completed. (Loss: {train_loss}, "
        f"total_train_time={total_train_time:.6f}s, "
        f"init_time={init_time:.6f}s, "
        f"loop_time={loop_time:.6f}s, "
        f"input_tokens={input_tokens_seen}, "
        f"supervised_tokens={supervised_tokens_seen}, "
        f"configured_local_steps="
        f"{int(_LOCAL_STATE.get('local_steps', 0) or 0)}, "
        f"executed_local_steps={executed_steps})"
    )

    upload_t0, upload_start_time = phase_start("upload_start")
    _probe_event(context, server_round, "lora_serialization_start")

    # For LoRA-based LLMs, only send adapter weights back to the server.
    # For non-LoRA models, fall back to the full state dict as before.
    net = _LOCAL_STATE["net"]
    if is_lora_model(net):
        arrays_out = get_peft_model_state_dict(net)
        logger.info("[TRA] Detected LoRA model – sending adapter-only state dict to server.")
    else:
        arrays_out = net.state_dict()
        logger.info("[TRA] Non-LoRA model – sending full state dict to server.")

    # Optional upload throttling for controlled communication experiments.
    # Applied only to selected device types.
    throttle_device = str(context.run_config.get("upload-throttle-device", "none"))
    throttle_mbps = float(context.run_config.get("upload-throttle-mbps", 0.0))

    current_device_name = str(_LOCAL_STATE.get("device_name", ""))
    current_device_uid = str(_LOCAL_STATE.get("device_uid", ""))

    enable_throttle = (
        throttle_mbps > 0
        and throttle_device != "none"
        and throttle_device in (current_device_name, current_device_uid)
    )

    if enable_throttle:
        import pickle
        payload_bytes = len(pickle.dumps(arrays_out, protocol=pickle.HIGHEST_PROTOCOL))
        throttle_sleep_time = payload_bytes * 8.0 / (throttle_mbps * 1e6)

        logger.info(
            f"[NetworkThrottle] enabled for {current_device_name}/{current_device_uid}: "
            f"target={throttle_mbps:.2f} Mbps, "
            f"payload={payload_bytes} bytes, "
            f"sleep={throttle_sleep_time:.6f}s"
        )

        time.sleep(throttle_sleep_time)
    else:
        payload_bytes = 0
        throttle_sleep_time = 0.0

    model_record = ArrayRecord(arrays_out)
    _probe_event(context, server_round, "lora_serialization_end")
    upload_elapsed_time = time.perf_counter() - upload_t0
    logger.info(
        f"[NetworkThrottle] upload_elapsed_time={upload_elapsed_time:.6f}s "
        f"(includes adapter extraction, optional throttle sleep, and ArrayRecord creation)"
    )

    logger.info(f"[train] Preparing metrics: lora_rank={_LOCAL_STATE.get('lora_rank')}, device_code={device_lookup[_LOCAL_STATE['device_name']]}")

    lora_rank_value = _LOCAL_STATE.get("lora_rank")
    if lora_rank_value is None:
        lora_rank_value = 0

    metrics = {
        "num-examples": int(executed_examples),
        "dataset_size": int(_LOCAL_STATE.get("dataset_size", 0)),
        "partition_id": int(_LOCAL_STATE.get("partition_id", -1)),
        "configured_num_examples": int(num_examples_used),
        "train_loss": float(train_loss),
        "train_energy": float(train_power),
        "train_sent": int(train_sent),
        "train_recv": int(train_recv),
        "train_time": float(total_train_time),
        "input_tokens": int(input_tokens_seen),
        "supervised_tokens": int(supervised_tokens_seen),
        "train_init_time": float(init_time),
        "train_loop_time": float(loop_time),
        "update_energy": float(update_power),
        "update_sent": int(update_sent),
        "update_recv": int(update_recv),
        "update_end_time": float(update_end_time),
        "upload_start_time": float(upload_start_time),
        "flops": float(get_flops(_LOCAL_STATE["net"], _LOCAL_STATE["device"])),
        "device_code": int(device_lookup[_LOCAL_STATE["device_name"]]),
        "lora_rank": int(lora_rank_value),
        "device_name_chars": [ord(c) for c in str(_LOCAL_STATE["device_name"])],
        "device_uid_chars": [ord(c) for c in str(_LOCAL_STATE.get("device_uid", "unknown"))],
        "upload_throttle_mbps": float(throttle_mbps if enable_throttle else 0.0),
        "upload_throttle_payload_bytes": int(payload_bytes),
        "upload_throttle_sleep_time": float(throttle_sleep_time),
        "client_upload_elapsed_time": float(upload_elapsed_time),
        "local_steps": int(executed_steps),
        "configured_local_steps": int(
            _LOCAL_STATE.get("local_steps", 0) or 0
        ),
    }
    if exposure_tracking_enabled:
        if len(visited_sample_ids) != int(executed_examples):
            raise RuntimeError("actual visited sample IDs differ from executed examples")
        metrics["visited_sample_ids"] = [int(value) for value in visited_sample_ids]
        metrics["unique_samples_round"] = len(set(visited_sample_ids))
        metrics["eligible_unique_samples"] = len(base_trainloader.dataset)
        if str(context.run_config.get("sample-selection-mode", "prefix")) == "exposure_aware":
            exposure_dir = Path(_LOCAL_PATH) / "exposure_state" / _safe_config_suffix(device_uid)
            exposure_dir.mkdir(parents=True, exist_ok=True)
            exposure_record = {
                "round": int(server_round),
                "eligible_unique_samples": len(base_trainloader.dataset),
                "visited_sample_ids": [int(value) for value in visited_sample_ids],
                "executed_examples": int(executed_examples),
                "executed_steps": int(executed_steps),
            }
            record_path = exposure_dir / f"round_{server_round:04d}.json"
            temp_path = record_path.with_suffix(".json.tmp")
            temp_path.write_text(json.dumps(exposure_record, sort_keys=True) + "\n")
            os.replace(temp_path, record_path)
    if _LOCAL_STATE.get("experiment_id"):
        metrics["experiment_id_chars"] = [ord(c) for c in _LOCAL_STATE["experiment_id"]]
    if _LOCAL_STATE.get("partition_manifest_sha256"):
        metrics["manifest_sha256_chars"] = [ord(c) for c in _LOCAL_STATE["partition_manifest_sha256"]]
    for k, v in metrics.items():
        if not isinstance(v, (int, float, list)):
            raise TypeError(f"Invalid train metric type: {k}={v!r}, type={type(v)}")
        if isinstance(v, list) and not all(isinstance(x, (int, float)) for x in v):
            raise TypeError(f"Invalid train metric list: {k}={v!r}")
    metric_record = MetricRecord(metrics)
    config_rec = msg.content.get("config", {})
    content = RecordDict({
        "arrays": model_record,
        "metrics": metric_record,
    })
    _probe_event(context, server_round, "handoff_to_flower")
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
        "num-examples": int(num_examples),
        "eval_loss": float(eval_loss),
        "eval_accuracy": float(eval_acc),
        "upload_energy": float(upload_power),
        "upload_sent": int(upload_sent),
        "upload_recv": int(upload_recv),
        "upload_end_time": float(upload_end_time),
        "eval_energy": float(eval_power),
        "eval_sent": int(eval_sent),
        "eval_recv": int(eval_recv),
        "eval_time": float(evaluate_end_time - eval_t0),
        "device_code": int(device_lookup[_LOCAL_STATE["device_name"]]),
        "device_name_chars": [ord(c) for c in str(_LOCAL_STATE["device_name"])],
        "device_uid_chars": [ord(c) for c in str(_LOCAL_STATE.get("device_uid", "unknown"))],
        "update_start_time": float(update_start_time),
    }
    for k, v in metrics.items():
        if not isinstance(v, (int, float, list)):
            raise TypeError(f"Invalid evaluate metric type: {k}={v!r}, type={type(v)}")
        if isinstance(v, list) and not all(isinstance(x, (int, float)) for x in v):
            raise TypeError(f"Invalid evaluate metric list: {k}={v!r}")
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
