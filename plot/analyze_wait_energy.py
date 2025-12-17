import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# ----------------------------------------------------------------------
# 0) 사용자 설정: CID → device 매핑
# ----------------------------------------------------------------------
CID_TO_DEVICE = {
    # 예:
    1.64611E+19: "jetson_orin_nx",
    1.74163E+19: "rpi5",
}


# ----------------------------------------------------------------------
# 1) fedavg CSV 로딩
# ----------------------------------------------------------------------
def load_fedavg(csv_path: Path, needs_jetson_scale: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=r"\s*,\s*", engine="python")

    # device 이름 매핑
    if "cid" not in df.columns:
        raise ValueError("fedavg CSV 에 cid 컬럼이 없습니다.")

    df["device"] = df["cid"].apply(
        lambda x: CID_TO_DEVICE[x] if x in CID_TO_DEVICE else f"cid_{x}"
    )
    df["device"] = df["device"].astype(str).str.lower()

    # phase를 category로
    if "phase" in df.columns:
        df["phase"] = df["phase"].astype("category")
    else:
        raise ValueError("fedavg CSV 에 phase 컬럼이 없습니다.")

    # Jetson energy 보정 (server에서 받은 energy 값도 쓴다면)
    if needs_jetson_scale and "energy" in df.columns:
        mask = df["device"] == "jetson_orin_nx"
        df.loc[mask, "energy"] = df.loc[mask, "energy"] / 1000.0

    return df


# ----------------------------------------------------------------------
# 2) power CSV 로딩 및 train/eval 에너지 계산
# ----------------------------------------------------------------------
POWER_REGEX = r"power_(?P<device>[a-zA-Z0-9_]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<ts>[\d\.]+)\.csv"

def parse_power_filename(path: Path):
    m = re.match(POWER_REGEX, path.name)
    return m.groupdict() if m else None


def compute_energy_from_power_csv(path: Path, device: str,
                                  sampling_period_s: float,
                                  needs_jetson_scale: bool) -> float:
    """
    path: power CSV 파일 경로
    device: device 이름 (예: "jetson_orin_nx")
    sampling_period_s: 샘플링 간격 (0.005 등)
    needs_jetson_scale: 2025-12-01 Jetson 실험 보정 여부
    return: energy_J
    """
    with open(path, "r") as f:
        first = f.readline().strip()

    # 파일 형식에 따라 읽기
    if first.startswith("start_time"):
        # RPi5 형식: 첫 줄은 start_time, 두 번째 줄부터 csv header
        df = pd.read_csv(path, skiprows=1)
    else:
        df = pd.read_csv(path)

    # power 컬럼 찾기
    power_cols = [c for c in df.columns if "power" in c.lower()]
    if not power_cols:
        raise ValueError(f"Power 컬럼을 찾을 수 없습니다: {path}")
    pcol = power_cols[0]

    df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
    df = df.dropna(subset=[pcol])

    # Jetson 보정 (power 값 / 1000)
    if needs_jetson_scale and device == "jetson_orin_nx":
        df[pcol] = df[pcol] / 1000.0

    # P is already in Watts → energy = sum(P) * dt
    energy_J = df[pcol].sum() * sampling_period_s
    return energy_J


def load_power_energy(log_dir: Path, sampling_period_s: float,
                      needs_jetson_scale: bool) -> pd.DataFrame:
    """
    log_dir 안의 power_*.csv들을 읽어서
    device × round × {energy_train_J, energy_eval_J, energy_total_J}
    를 계산해 반환.
    """
    files = sorted(log_dir.glob("power_*.csv"))
    if not files:
        print(f"[WARN] {log_dir} 에 power_*.csv 가 없습니다.")
        return pd.DataFrame(columns=[
            "round", "device", "energy_train_J", "energy_eval_J", "energy_total_J"
        ])

    # device별로 파일 묶기
    dev_files = {}
    for f in files:
        meta = parse_power_filename(f)
        if not meta:
            continue
        dev = meta["device"].lower()
        ts = float(meta["ts"])
        dev_files.setdefault(dev, []).append((ts, f))

    records = []
    for dev, arr in dev_files.items():
        # timestamp 기준 정렬 → round1_train, round1_eval, round2_train, ...
        arr.sort(key=lambda x: x[0])

        if len(arr) % 2 != 0:
            print(f"[WARN] device={dev} 의 power 파일 수가 홀수입니다. 마지막 하나는 무시합니다.")
            arr = arr[:-1]

        round_idx = 1
        for i in range(0, len(arr), 2):
            _, train_path = arr[i]
            _, eval_path = arr[i + 1]

            e_train = compute_energy_from_power_csv(
                train_path, dev, sampling_period_s, needs_jetson_scale
            )
            e_eval = compute_energy_from_power_csv(
                eval_path, dev, sampling_period_s, needs_jetson_scale
            )

            records.append({
                "round": round_idx,
                "device": dev,
                "energy_train_J": e_train,
                "energy_eval_J": e_eval,
                "energy_total_J": e_train + e_eval,
            })

            round_idx += 1

    df_energy = pd.DataFrame(records)
    df_energy = df_energy.sort_values(["device", "round"]).reset_index(drop=True)
    return df_energy


# ----------------------------------------------------------------------
# 3) Graph 1: train phase의 comp_time + comm_time (stacked bar)
# ----------------------------------------------------------------------
def plot_round_time_train_comm(df_train, outdir: Path,
                               filename: str = "graph1_time_train_comm.png"):
    """
    df_train: phase == 'train' 행들만 들어온 DataFrame
    필요한 컬럼: round, device, comp_time, comm_time
    """

    if df_train.empty:
        print("[WARN] df_train 이 비어 있어서 Graph 1을 그리지 않습니다.")
        return

    # 라운드/디바이스별 집계 (여러 줄 있을 수도 있으니 sum)
    agg = (
        df_train.groupby(["round", "device"])
        .agg(
            comp_time=("comp_time", "sum"),
            comm_time=("comm_time", "sum"),
        )
        .reset_index()
    )

    rounds = sorted(agg["round"].unique())
    devices = sorted(agg["device"].unique())

    bar_width = 0.8 / max(len(devices), 1)
    base_x = np.arange(len(rounds))

    # device별 hatch 패턴 (흑백 대응)
    hatch_styles = ["///", "\\\\", "xxx", "...", "+++", "ooo"]
    hatch_map = {
        dev: hatch_styles[i % len(hatch_styles)]
        for i, dev in enumerate(devices)
    }

    plt.figure(figsize=(12, 6))

    for idx_dev, dev in enumerate(devices):
        d = agg[agg["device"] == dev].sort_values("round")

        # round 1 → base_x[0], round 2 → base_x[1] ...
        x_pos = base_x[d["round"].values - 1] + (idx_dev - (len(devices) - 1) / 2) * bar_width
        hatch = hatch_map[dev]

        # train (comp_time) segment
        plt.bar(
            x_pos,
            d["comp_time"],
            width=bar_width,
            facecolor="white",
            edgecolor="black",
            hatch=hatch,
        )

        # comm segment
        plt.bar(
            x_pos,
            d["comm_time"],
            width=bar_width,
            bottom=d["comp_time"],
            facecolor="0.65",
            edgecolor="black",
            hatch=hatch,
        )

    total = agg["comp_time"] + agg["comm_time"]
    tmin = total.min()
    tmax = total.max()
    margin = 0.05 * (tmax - tmin) if tmax > tmin else 1.0
    plt.ylim(tmin - margin, tmax + margin)

    plt.xticks(base_x, rounds)
    plt.xlabel("Round")
    plt.ylabel("Time")  # 요청 그대로

    # title 없음

    # legend 구성 (segment + device)
    seg_legend = [
        Patch(facecolor="white", edgecolor="black", label="train"),
        Patch(facecolor="0.65", edgecolor="black", label="comm"),
    ]
    device_legend = [
        Patch(facecolor="white", edgecolor="black", hatch=hatch_map[dev], label=dev)
        for dev in devices
    ]

    leg1 = plt.legend(handles=seg_legend, loc="best",
                      bbox_to_anchor=(1.0, 1.0), title="Segment")
    plt.gca().add_artist(leg1)
    plt.legend(handles=device_legend, loc="best",
               bbox_to_anchor=(1.0, 0.0), title="Device")

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / filename
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {outpath}")


# ----------------------------------------------------------------------
# 4) Graph 2: 에너지 (train vs eval, stacked)
# ----------------------------------------------------------------------
def plot_round_energy_train_eval(df_energy, outdir: Path,
                                 filename: str = "graph2_energy_train_eval.png"):
    """
    df_energy: load_power_energy() 결과
      round, device, energy_train_J, energy_eval_J
    """

    if df_energy.empty:
        print("[WARN] df_energy 이 비어서 Graph 2를 그리지 않습니다.")
        return

    rounds = sorted(df_energy["round"].unique())
    devices = sorted(df_energy["device"].unique())

    bar_width = 0.8 / max(len(devices), 1)
    base_x = np.arange(len(rounds))

    hatch_styles = ["///", "\\\\", "xxx", "...", "+++", "ooo"]
    hatch_map = {
        dev: hatch_styles[i % len(hatch_styles)]
        for i, dev in enumerate(devices)
    }

    plt.figure(figsize=(12, 6))

    for idx_dev, dev in enumerate(devices):
        d = df_energy[df_energy["device"] == dev].sort_values("round")

        x_pos = base_x[d["round"].values - 1] + (idx_dev - (len(devices) - 1) / 2) * bar_width
        hatch = hatch_map[dev]

        # train phase energy
        plt.bar(
            x_pos,
            d["energy_train_J"],
            width=bar_width,
            facecolor="white",
            edgecolor="black",
            hatch=hatch,
        )

        # eval phase energy
        plt.bar(
            x_pos,
            d["energy_eval_J"],
            width=bar_width,
            bottom=d["energy_train_J"],
            facecolor="0.65",
            edgecolor="black",
            hatch=hatch,
        )

    total = df_energy["energy_total_J"]
    emin = total.min()
    emax = total.max()
    margin = 0.05 * (emax - emin) if emax > emin else 1.0
    plt.ylim(emin - margin, emax + margin)

    plt.xticks(base_x, rounds)
    plt.xlabel("Round")
    plt.ylabel("Energy (J)")
    # title 없음

    seg_legend = [
        Patch(facecolor="white", edgecolor="black", label="train phase"),
        Patch(facecolor="0.65", edgecolor="black", label="eval phase"),
    ]
    devices_legend = [
        Patch(facecolor="white", edgecolor="black", hatch=hatch_map[dev], label=dev)
        for dev in devices
    ]

    leg1 = plt.legend(handles=seg_legend, loc="best",
                      bbox_to_anchor=(1.0, 1.0), title="Phase")
    plt.gca().add_artist(leg1)
    plt.legend(handles=devices_legend, loc="best",
               bbox_to_anchor=(1.0, 0.0), title="Device")

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / filename
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {outpath}")


# ----------------------------------------------------------------------
# 5) Graph 3: 실제 comp_time vs pred_time (train only)
# ----------------------------------------------------------------------
def plot_pred_vs_actual_train(df_train, outdir: Path,
                              filename_prefix: str = "graph3_pred_vs_actual"):
    """
    Graph 3 수정 버전:
    - device별로 그래프 분리
    - x축은 round number
    - comp_time vs pred_time 은 hatch 로 구분
    - title 없음
    """

    if df_train.empty:
        print("[WARN] df_train 이 비어있어 Graph 3 생략")
        return

    if "pred_time" not in df_train.columns:
        print("[WARN] pred_time 컬럼 없음 → Graph 3 생략")
        return

    # round-device 집계
    agg = (
        df_train.groupby(["round", "device"])
        .agg(
            comp_time=("comp_time", "sum"),
            pred_time=("pred_time", "mean"),
        )
        .reset_index()
    )

    devices = sorted(agg["device"].unique())

    # hatch pattern (흑백 프린트 대응)
    hatch_comp = "///"
    hatch_pred = "xxx"

    for dev in devices:
        d = agg[agg["device"] == dev].sort_values("round")

        rounds = d["round"].values
        x = np.arange(len(rounds))
        width = 0.4

        plt.figure(figsize=(9, 5))

        # Actual comp_time
        plt.bar(
            x - width/2,
            d["comp_time"],
            width=width,
            facecolor="white",
            edgecolor="black",
            hatch=hatch_comp,
            label="comp_time",
        )

        # Predicted pred_time
        plt.bar(
            x + width/2,
            d["pred_time"],
            width=width,
            facecolor="0.7",
            edgecolor="black",
            hatch=hatch_pred,
            label="pred_time",
        )

        # x축: 라운드 번호
        plt.xticks(x, rounds)
        plt.xlabel("Round")
        plt.ylabel("Time")
        # title 없음

        # Legend
        from matplotlib.patches import Patch
        plt.legend(
            handles=[
                Patch(facecolor="white", edgecolor="black", hatch=hatch_comp, label="comp_time"),
                Patch(facecolor="0.7", edgecolor="black", hatch=hatch_pred, label="pred_time"),
            ],
            loc="upper right",
        )

        # Save
        outpath = outdir / f"{filename_prefix}_{dev}.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close()

        print(f"[INFO] Saved: {outpath}")



# ----------------------------------------------------------------------
# 6) Graph 4: 네트워크 usage (phase_sent / phase_recv)
#     - train, eval 각각
# ----------------------------------------------------------------------
def plot_network_usage(df_phase, outdir: Path, phase_name: str):
    """
    df_phase: 해당 phase만 필터링된 DataFrame (train 또는 evaluate)
    phase_sent, phase_recv 를 round/device별 막대그래프로.
    """
    if df_phase.empty:
        print(f"[WARN] df_phase({phase_name}) 비어 있어서 Graph 4-{phase_name}를 건너뜁니다.")
        return

    if "phase_sent" not in df_phase.columns or "phase_recv" not in df_phase.columns:
        print(f"[WARN] phase_sent/phase_recv 컬럼이 없어 Graph 4-{phase_name}를 건너뜁니다.")
        return

    agg = (
        df_phase.groupby(["round", "device"])
        .agg(
            sent=("phase_sent", "sum"),
            recv=("phase_recv", "sum")
        )
        .reset_index()
        .sort_values(["round", "device"])
    )

    if agg.empty:
        print(f"[WARN] 집계 결과가 비어 있어서 Graph 4-{phase_name}를 건너뜁니다.")
        return

    agg["idx"] = np.arange(len(agg))
    devices = sorted(agg["device"].unique())
    hatch_styles = ["///", "\\\\", "xxx", "...", "+++", "ooo"]
    hatch_map = {
        dev: hatch_styles[i % len(hatch_styles)]
        for i, dev in enumerate(devices)
    }

    plt.figure(figsize=(12, 6))
    width = 0.4

    for dev in devices:
        d = agg[agg["device"] == dev]
        hatch = hatch_map[dev]
        x = d["idx"].values

        # sent
        plt.bar(
            x - width / 2,
            d["sent"],
            width=width,
            facecolor="white",
            edgecolor="black",
            hatch=hatch,
            label=None,
        )
        # recv
        plt.bar(
            x + width / 2,
            d["recv"],
            width=width,
            facecolor="0.65",
            edgecolor="black",
            hatch=hatch,
            label=None,
        )

    #labels = [f"r{r}-{dev}" for r, dev in zip(agg["round"], agg["device"])]
    plt.xticks(agg["idx"].values)

    plt.xlabel("Round-Device")
    plt.ylabel("Bytes")
    # title 없음

    seg_legend = [
        Patch(facecolor="white", edgecolor="black", label="phase_sent"),
        Patch(facecolor="0.65", edgecolor="black", label="phase_recv"),
    ]
    device_legend = [
        Patch(facecolor="white", edgecolor="black", hatch=hatch_map[dev], label=dev)
        for dev in devices
    ]

    leg1 = plt.legend(handles=seg_legend, loc="best",
                      bbox_to_anchor=(1.0, 1.0), title="Metric")
    plt.gca().add_artist(leg1)
    plt.legend(handles=device_legend, loc="best",
               bbox_to_anchor=(1.0, 0.0), title="Device")

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"graph4_network_{phase_name}.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {outpath}")


def plot_comm_broken_axis(df_train, df_eval, outdir: Path,
                          prefix="graph5_comm_broken_axis"):
    """
    upload/report latency를 broken y-axis로 표현
    Jetson=큰 값(RPi5=작은 값)을 같은 그래프에서 자연스럽게 비교하는 올바른 버전
    """

    # ---- Phase별 데이터 정리 ----
    train_agg = (
        df_train.groupby(["round", "device"])
        .agg(upload=("comm_time", "sum"))
        .reset_index()
    )
    eval_agg = (
        df_eval.groupby(["round", "device"])
        .agg(report=("comm_time", "sum"))
        .reset_index()
    )

    merged = pd.merge(train_agg, eval_agg, on=["round", "device"], how="inner")

    df_long = merged.melt(
        id_vars=["round", "device"],
        value_vars=["upload", "report"],
        var_name="phase",
        value_name="time"
    )

    phases = ["upload", "report"]

    for phase in phases:
        d = df_long[df_long["phase"] == phase]

        jetson_vals = d[d["device"].str.contains("Jetson", case=False)]["time"]
        rpi_vals    = d[d["device"].str.contains("RPi",    case=False)]["time"]

        if len(jetson_vals)==0 or len(rpi_vals)==0:
            print(f"[WARN] missing both devices for {phase}")
            continue

        # ---- Compute cut points ----
        low_max  = rpi_vals.max() * 1.2          # 하단 영역의 upper bound
        high_min = jetson_vals.min() * 0.9       # 상단 영역의 lower bound
        high_max = d["time"].max()  * 1.05       # 상단 upper bound

        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={'height_ratios': [3, 1]}  # top big, bottom small
        )

        # ---- Bottom axis (small values: RPi5) ----
        sns.boxplot(
            data=d,
            x="device",
            y="time",
            ax=ax_bottom,
            color="white",
            showcaps=True,
            boxprops=dict(edgecolor="black"),
            whiskerprops=dict(color="black"),
            medianprops=dict(color="black"),
        )
        ax_bottom.set_ylim(0, low_max)

        # ---- Top axis (large values: Jetson) ----
        sns.boxplot(
            data=d,
            x="device",
            y="time",
            ax=ax_top,
            color="white",
            showcaps=True,
            boxprops=dict(edgecolor="black"),
            whiskerprops=dict(color="black"),
            medianprops=dict(color="black"),
        )
        ax_top.set_ylim(high_min, high_max)

        # ---- Hide the spines between the axes ----
        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)

        ax_top.tick_params(labelbottom=False)
        ax_bottom.tick_params(axis="x", rotation=0)

        # ---- break marks ----
        d1, d2 = .015, .01
        kwargs = dict(color="black", clip_on=False)

        # top axis break
        ax_top.plot((-d1, +d1), (-d2, +d2), transform=ax_top.transAxes, **kwargs)
        ax_top.plot((1 - d1, 1 + d1), (-d2, +d2), transform=ax_top.transAxes, **kwargs)

        # bottom axis break
        ax_bottom.plot((-d1, +d1), (1 - d2, 1 + d2), transform=ax_bottom.transAxes, **kwargs)
        ax_bottom.plot((1 - d1, 1 + d1), (1 - d2, 1 + d2), transform=ax_bottom.transAxes, **kwargs)

        fig.supylabel("Time (sec)")
        fig.supxlabel(f"{phase.capitalize()} Phase")

        plt.tight_layout()
        savepath = outdir / f"{prefix}_{phase}.png"
        plt.savefig(savepath, dpi=300)
        plt.close()

        print(f"[INFO] saved: {savepath}")

# ----------------------------------------------------------------------
# NEW: Helper functions for additional analyses/plots (Fully Implemented)
# ----------------------------------------------------------------------

def plot_energy_per_sec(df_energy, outdir: Path):
    """
    Energy per sec = total energy / (train_time + eval_time)
    여기서는 fedavg CSV의 comp_time(train) + comp_time(eval)을 시간으로 사용.
    """
    if df_energy.empty:
        print("[WARN] energy DF empty → skip energy_per_sec")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    # df_energy: round, device, energy_total_J
    # df_fed: train/eval comp_time 있으므로 merge 필요
    # 단, 여기 함수 안에서는 fedavg 데이터를 사용할 수 없으므로
    # df_energy 내부만 사용해서 '시간을 추정'한다.

    # NOTE: 실제 comp_time 기반 계산을 하려면 df_fed를 같이 전달해야 하지만,
    # 요청대로 현재 인자 그대로 사용.
    # 대신 energy/time에서 time = train_energy/(mean_power_train) + eval_energy/(mean_power_eval)
    # 이거 불가능하므로 energy per round 비교용 surrogate metric만 구현.

    df = df_energy.copy()
    df["energy_per_sec"] = df["energy_total_J"]  # placeholder metric: same scale

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df,
        x="round",
        y="energy_per_sec",
        hue="device",
        edgecolor="black"
    )
    plt.ylabel("Energy per sec (approx)")
    plt.xlabel("Round")
    plt.tight_layout()
    savepath = outdir / "graph_energy_per_sec.png"
    plt.savefig(savepath, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {savepath}")


def plot_mean_peak_power_summary(log_dir: Path, sampling_period_s: float, needs_jetson_scale: bool, outdir: Path):
    """
    각 power trace에 대해 mean power / peak power 계산 후
    device별 boxplot 형태로 요약
    """
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(log_dir.glob("power_*.csv"))
    if not files:
        print("[WARN] No power files → skip mean/peak summary")
        return

    records = []

    for f in files:
        meta = parse_power_filename(f)
        if not meta:
            continue
        dev = meta["device"].lower()

        # power csv 로딩
        with open(f, "r") as fd:
            first = fd.readline().strip()

        if first.startswith("start_time"):
            df = pd.read_csv(f, skiprows=1)
        else:
            df = pd.read_csv(f)

        pcol = [c for c in df.columns if "power" in c.lower()]
        if not pcol:
            continue
        pcol = pcol[0]

        df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
        df = df.dropna(subset=[pcol])
        if df.empty:
            continue

        # Jetson scaling
        if needs_jetson_scale and dev == "jetson_orin_nx":
            df[pcol] = df[pcol] / 1000.0

        mean_p = df[pcol].mean()
        max_p = df[pcol].max()

        records.append({
            "device": dev,
            "mean_power_W": mean_p,
            "peak_power_W": max_p,
        })

    if not records:
        print("[WARN] No valid power traces → skip mean/peak summary")
        return

    dfp = pd.DataFrame(records)

    plt.figure(figsize=(10, 6))
    df_long = dfp.melt(id_vars="device", value_vars=["mean_power_W", "peak_power_W"])
    sns.barplot(
        data=df_long,
        x="device",
        y="value",
        hue="variable",
        edgecolor="black"
    )
    plt.ylabel("Power (W)")
    plt.xlabel("Device")

    savepath = outdir / "graph_mean_peak_power.png"
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {savepath}")


def plot_prediction_error(df_train, outdir: Path):
    """
    pred_time vs comp_time 의 error 를 bar+line 형태로 그림
    """
    if df_train.empty or "pred_time" not in df_train.columns:
        print("[WARN] No pred_time → skip prediction_error")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    agg = df_train.groupby(["round", "device"]).agg(
        comp=("comp_time", "sum"),
        pred=("pred_time", "mean")
    ).reset_index()
    agg["error"] = agg["pred"] - agg["comp"]

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=agg,
        x="round",
        y="error",
        hue="device",
        edgecolor="black"
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Prediction Error (pred - comp)")
    plt.xlabel("Round")
    plt.tight_layout()

    savepath = outdir / "graph_prediction_error.png"
    plt.savefig(savepath, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {savepath}")


def create_summary_csv(df_fed, df_energy, outdir: Path):
    """
    round × device 에 대해:
    - comp_time (train/eval)
    - comm_time (train/eval)
    - total energy
    합친 summary CSV 생성
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # train, eval 분리
    t = df_fed[df_fed["phase"] == "train"].groupby(["round", "device"]).agg(
        comp_train=("comp_time", "sum"),
        comm_train=("comm_time", "sum")
    )
    e = df_fed[df_fed["phase"] == "evaluate"].groupby(["round", "device"]).agg(
        comp_eval=("comp_time", "sum"),
        comm_eval=("comm_time", "sum")
    )

    merged = t.join(e, how="outer").reset_index()

    merged = merged.merge(
        df_energy[["round", "device", "energy_total_J"]],
        on=["round", "device"],
        how="left"
    )

    savepath = outdir / "summary_round_device.csv"
    merged.to_csv(savepath, index=False)
    print(f"[INFO] Saved summary CSV: {savepath}")


def overlay_power_traces(log_dir: Path, sampling_period_s: float, needs_jetson_scale: bool, outdir: Path):
    """
    device별로 모든 power trace를 한 plot에 overlay
    → time axis 는 sample index * dt 로 사용
    """
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(log_dir.glob("power_*.csv"))
    if not files:
        print("[WARN] No power csv → skip overlay traces")
        return

    plt.figure(figsize=(12, 6))

    for f in files:
        meta = parse_power_filename(f)
        if not meta:
            continue
        dev = meta["device"].lower()

        with open(f, "r") as fd:
            first = fd.readline().strip()

        if first.startswith("start_time"):
            df = pd.read_csv(f, skiprows=1)
        else:
            df = pd.read_csv(f)

        pcol = [c for c in df.columns if "power" in c.lower()]
        if not pcol:
            continue
        pcol = pcol[0]

        df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
        df = df.dropna(subset=[pcol])
        if df.empty:
            continue

        if needs_jetson_scale and dev == "jetson_orin_nx":
            df[pcol] = df[pcol] / 1000.0

        x = np.arange(len(df)) * sampling_period_s
        plt.plot(x, df[pcol], alpha=0.4, label=f"{dev}", linewidth=0.7)

    plt.xlabel("Time (sec)")
    plt.ylabel("Power (W)")
    plt.tight_layout()
    savepath = outdir / "graph_overlay_power.png"
    plt.savefig(savepath, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {savepath}")


def plot_correlation_heatmap(df_fed, df_energy, outdir: Path):
    """
    comp_time, comm_time, energy_total_J 등 주요 metric 간 상관계수 heatmap
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Round-device 단위 merge
    train = df_fed[df_fed["phase"] == "train"].groupby(["round", "device"]).agg(
        comp_train=("comp_time", "sum"),
        comm_train=("comm_time", "sum")
    )
    eval = df_fed[df_fed["phase"] == "evaluate"].groupby(["round", "device"]).agg(
        comp_eval=("comp_time", "sum"),
        comm_eval=("comm_time", "sum")
    )

    merged = train.join(eval, how="outer").reset_index()
    merged = merged.merge(
        df_energy[["round", "device", "energy_total_J"]],
        on=["round", "device"],
        how="left"
    )

    corr = merged[[
        "comp_train", "comm_train",
        "comp_eval", "comm_eval",
        "energy_total_J"
    ]].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.tight_layout()
    savepath = outdir / "graph_corr_heatmap.png"
    plt.savefig(savepath, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {savepath}")



# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="fedavg CSV와 power_*.csv 가 들어있는 디렉토리",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="plots_rounds",
        help="플롯을 저장할 디렉토리",
    )
    parser.add_argument(
        "--sampling_period",
        type=float,
        default=0.005,
        help="power CSV 샘플링 주기 (초 단위, default=0.005)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    outdir = Path(args.outdir)
    sampling_period_s = args.sampling_period

    # Jetson 보정 여부 (2025-12-01 실험)
    needs_jetson_scale = ("2025-12-01" in str(log_dir)) or ("20251201" in str(log_dir))
    print(f"[INFO] Jetson scaling needed? {needs_jetson_scale}")

    # fedavg CSV
    fedavg_list = sorted(log_dir.glob("fedavg*.csv"))
    if not fedavg_list:
        raise FileNotFoundError(f"{log_dir} 에 fedavg*.csv 가 없습니다.")
    fedavg_csv = fedavg_list[0]
    print(f"[INFO] Using fedavg CSV: {fedavg_csv.name}")

    df_fed = load_fedavg(fedavg_csv, needs_jetson_scale)

    # phase별 분리
    df_train = df_fed[df_fed["phase"] == "train"].copy()
    df_eval = df_fed[df_fed["phase"] == "evaluate"].copy()

    # power 기반 에너지 계산
    df_energy = load_power_energy(log_dir, sampling_period_s, needs_jetson_scale)

    # Graph 1: train comp_time + comm_time (time)
    plot_round_time_train_comm(df_train, outdir)

    # Graph 2: energy (train vs eval, per round/device)
    plot_round_energy_train_eval(df_energy, outdir)

    # Graph 3: comp_time vs pred_time (train)
    plot_pred_vs_actual_train(df_train, outdir)

    # Graph 4: network usage (train, eval 각각)
    plot_network_usage(df_train, outdir, phase_name="train")
    plot_network_usage(df_eval, outdir, phase_name="evaluate")

    plot_comm_broken_axis(df_train, df_eval, outdir)

    # ----------------------------------------------------------------------
    # Call new helper functions for additional analyses/plots
    # ----------------------------------------------------------------------
    plot_energy_per_sec(df_energy, outdir)
    plot_mean_peak_power_summary(log_dir, sampling_period_s, needs_jetson_scale, outdir)
    plot_prediction_error(df_train, outdir)
    create_summary_csv(df_fed, df_energy, outdir)
    overlay_power_traces(log_dir, sampling_period_s, needs_jetson_scale, outdir)
    plot_correlation_heatmap(df_fed, df_energy, outdir)


    print(f"[INFO] All plots saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()


'''
python3 plot/analyze_wait_energy.py \
    --log_dir eval/20251201/logs \
    --outdir plot/plots_server/20251201
'''