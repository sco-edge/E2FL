import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Fast non-GUI backend
import matplotlib.pyplot as plt
import re


# -----------------------------------------------------
# 1) power 파일명 파싱: power_<device>_<date>_<ts>.csv
# -----------------------------------------------------
POWER_REGEX = r"power_(?P<device>[a-zA-Z0-9_]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<ts>[\d\.]+)\.csv"

def parse_power_filename(path: Path):
    m = re.match(POWER_REGEX, path.name, flags=re.IGNORECASE)
    if not m:
        return None
    return m.groupdict()


# -----------------------------------------------------
# 2) CSV 로더 (+ 필요 시 Jetson 스케일링)
# -----------------------------------------------------
def load_power_log(path: Path, meta, scale_jetson: bool):
    device = meta["device"]
    # date = meta["date"]  # 지금은 안 씀, 필요하면 살릴 수 있음

    with open(path, "r") as f:
        first = f.readline().strip()

    # Jetson 형식
    if first.startswith("Timestamp"):
        df = pd.read_csv(path)
        t = [c for c in df.columns if "Timestamp" in c][0]
        p = [c for c in df.columns if "Power" in c][0]

    # RPi5 형식
    elif first.startswith("start_time"):
        df = pd.read_csv(path, skiprows=1)
        t = [c for c in df.columns if "time" in c.lower()][0]
        p = [c for c in df.columns if "power" in c.lower()][0]

    # fallback
    else:
        df = pd.read_csv(path)
        t = [c for c in df.columns if "time" in c.lower()][0]
        p = [c for c in df.columns if "power" in c.lower()][0]

    df = df.rename(columns={t: "time_s", p: "power_W"})
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["power_W"] = pd.to_numeric(df["power_W"], errors="coerce")
    df = df.dropna()

    # ⭐ 폴더 기준으로 보정: 이 run에서 scale_jetson=True 인 경우에만
    if scale_jetson and device == "jetson_orin_nx":
        df["power_W"] = df["power_W"] / 1000.0

    return df.sort_values("time_s").reset_index(drop=True)


# -----------------------------------------------------
# 3) 에너지 계산 (적분)
# -----------------------------------------------------
def compute_energy(df: pd.DataFrame):
    df = df.copy()
    df["dt"] = df["time_s"].diff().fillna(0)
    energy_J = (df["power_W"] * df["dt"]).sum()
    duration = df["time_s"].iloc[-1] - df["time_s"].iloc[0]
    avg_power = energy_J / duration if duration > 0 else 0
    return energy_J, avg_power, duration  # J, W, s


# -----------------------------------------------------
# 4) plotting helpers
# -----------------------------------------------------
def plot_power_trace(df, outdir, device, label):
    plt.figure(figsize=(8, 4))
    plt.plot(df["time_s"], df["power_W"], linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.tight_layout()
    plt.savefig(outdir / f"trace_{device}_{label}.png")
    plt.close()


def plot_power_trace_smoothed(df, outdir, device, label, window_sec=0.5):
    df2 = df.copy()

    if len(df2) > 1:
        avg_dt = (df2["time_s"].iloc[-1] - df2["time_s"].iloc[0]) / (len(df2) - 1)
    else:
        avg_dt = 0.01

    w = max(int(window_sec / avg_dt), 1)
    df2["smooth"] = df2["power_W"].rolling(w, center=True).mean()

    plt.figure(figsize=(8, 4))
    plt.plot(df2["time_s"], df2["power_W"], color="gray", alpha=0.3,
             linewidth=0.6, label="raw")
    plt.plot(df2["time_s"], df2["smooth"], color="red",
             linewidth=1.0, label="smooth")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"trace_smoothed_{device}_{label}.png")
    plt.close()


def plot_overlay(device, df_list, outdir):
    plt.figure(figsize=(10, 4))

    for item in df_list:
        df = item["df"]
        r = item["round"]
        plt.plot(df["time_s"], df["power_W"], linewidth=0.6, label=f"r{r}")

    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"overlay_{device}.png")
    plt.close()


def plot_energy_bar(df_stats, outdir):
    plt.figure(figsize=(10, 4))

    rounds = sorted(df_stats["round"].unique())
    devices = list(df_stats["device"].unique())
    x = np.arange(len(rounds))
    width = 0.35 if len(devices) == 2 else 0.6 / max(len(devices), 1)

    for i, dev in enumerate(devices):
        df_d = df_stats[df_stats["device"] == dev]
        y = []
        for r in rounds:
            row = df_d[df_d["round"] == r]
            y.append(row["energy_total_J"].iloc[0] if not row.empty else 0.0)

        offset = (i - (len(devices) - 1) / 2) * width
        plt.bar(x + offset, y, width=width, label=dev, alpha=0.8)

    plt.xlabel("Round")
    plt.ylabel("Energy (J)")
    plt.xticks(x, rounds)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "round_energy.png")
    plt.close()

def plot_phase_energy_per_device(df_stats, outdir):
    """
    device별로 train/eval 에너지를 round별로 비교하는 bar 그래프 생성.
    예시 출력:
      phase_energy_jetson_orin_nx.png
      phase_energy_RPi5.png
    """
    devices = df_stats["device"].unique()

    for dev in devices:
        df_d = df_stats[df_stats["device"] == dev].sort_values("round")
        rounds = df_d["round"].tolist()

        train_vals = df_d["energy_train_J"].tolist()
        eval_vals = df_d["energy_eval_J"].tolist()

        x = np.arange(len(rounds))
        width = 0.35

        plt.figure(figsize=(10, 4))
        plt.bar(x - width/2, train_vals, width, label="train", color="#4C72B0")
        plt.bar(x + width/2, eval_vals, width, label="evaluate", color="#DD8452")

        plt.xlabel("Round")
        plt.ylabel("Energy (J)")
        plt.xticks(x, rounds)
        plt.legend(loc="best")
        plt.tight_layout()

        outpath = outdir / f"phase_energy_{dev}.png"
        plt.savefig(outpath)
        plt.close()
    
def plot_phase_energy_stacked(df_stats, outdir):
    """
    두 디바이스를 비교하는 stacked bar graph.
    - x축: round
    - 각 round마다 device별 bar 2개가 나란히 배치
    - bar 내부는 train(J) + eval(J) 으로 stacked
    """

    devices = sorted(df_stats["device"].unique())
    rounds = sorted(df_stats["round"].unique())
    x = np.arange(len(rounds))

    width = 0.35 if len(devices) == 2 else 0.6 / max(len(devices), 1)

    # 디바이스별 색 팔레트
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # 필요 시 확장 가능

    plt.figure(figsize=(12, 5))

    for i, dev in enumerate(devices):
        df_d = df_stats[df_stats["device"] == dev].sort_values("round")

        train_vals = df_d["energy_train_J"].tolist()
        eval_vals = df_d["energy_eval_J"].tolist()

        offset = (i - (len(devices)-1)/2) * width

        # Train (bottom)
        plt.bar(
            x + offset,
            train_vals,
            width=width,
            label=f"{dev} train",
            color=colors[i],
            alpha=0.8
        )

        # Evaluate (stacked on top)
        plt.bar(
            x + offset,
            eval_vals,
            width=width,
            bottom=train_vals,
            label=f"{dev} eval",
            color=colors[i],
            alpha=0.4,
            hatch="///"
        )

    plt.xlabel("Round")
    plt.ylabel("Energy (J)")
    plt.xticks(x, rounds)
    plt.legend(ncol=2, loc="best")
    plt.tight_layout()

    plt.savefig(outdir / "phase_energy_stacked.png")
    plt.close()

# -----------------------------------------------------
# 5) main analysis: device-aware + train+comm concat
# -----------------------------------------------------
def run_analysis(power_dir, outdir):
    power_dir = Path(power_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ⭐ 2025-12-01 폴더인지 확인하여 jetson 스케일링 여부 결정
    power_dir_str = str(power_dir)
    needs_jetson_scale = ("2025-12-01" in power_dir_str) or ("20251201" in power_dir_str)

    files = sorted(power_dir.glob("power_*.csv"))
    files_by_device = {}

    # group by device
    for f in files:
        meta = parse_power_filename(f)
        if meta is None:
            continue
        device = meta["device"]
        files_by_device.setdefault(device, []).append((f, meta))

    # sort each device's file list by ts token
    for dev in files_by_device:
        files_by_device[dev] = sorted(files_by_device[dev], key=lambda x: x[1]["ts"])

    stats = []
    overlay_map = {}

    for device, file_pairs in files_by_device.items():
        round_idx = 1

        # 파일 2개씩 = (train, evaluate)
        for i in range(0, len(file_pairs), 2):
            (f_train, meta_train) = file_pairs[i]
            f_eval, meta_eval = (None, None)
            if i + 1 < len(file_pairs):
                (f_eval, meta_eval) = file_pairs[i + 1]

            # train
            df_train = load_power_log(f_train, meta_train, needs_jetson_scale)
            E_train, avg_train, dur_train = compute_energy(df_train)

            # evaluate
            df_eval = None
            E_eval = avg_eval = dur_eval = 0.0
            if f_eval:
                df_eval = load_power_log(f_eval, meta_eval, needs_jetson_scale)
                E_eval, avg_eval, dur_eval = compute_energy(df_eval)

            # train + eval 이어붙이기
            if df_eval is not None:
                df_eval_shifted = df_eval.copy()
                df_eval_shifted["time_s"] += df_train["time_s"].iloc[-1] + 1e-6
                df_round = pd.concat([df_train, df_eval_shifted], ignore_index=True)
            else:
                df_round = df_train.copy()

            # stats
            stats.append({
                "round": round_idx,
                "device": device,
                "energy_total_J": E_train + E_eval,
                "energy_train_J": E_train,
                "energy_eval_J": E_eval,
                "duration_total_s": dur_train + dur_eval,
                "duration_train_s": dur_train,
                "duration_eval_s": dur_eval,
                "file_train": f_train.name,
                "file_eval": f_eval.name if f_eval else None
            })

            label_round = f"round{round_idx}"
            plot_power_trace(df_round, outdir, device, label_round)
            plot_power_trace_smoothed(df_round, outdir, device, label_round)

            overlay_map.setdefault(device, []).append({"round": round_idx, "df": df_round})

            round_idx += 1

    df_stats = pd.DataFrame(stats).sort_values(["device", "round"])
    df_stats.to_csv(outdir / "round_energy_stats.csv", index=False)

    plot_energy_bar(df_stats, outdir)
    plot_phase_energy_per_device(df_stats, outdir)
    plot_phase_energy_stacked(df_stats, outdir) 

    for dev, lst in overlay_map.items():
        plot_overlay(dev, lst, outdir)

    print("===== Analysis Complete =====")
    print(df_stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--power_dir", required=True)
    parser.add_argument("--outdir", default="power_out")
    args = parser.parse_args()
    run_analysis(args.power_dir, args.outdir)


if __name__ == "__main__":
    main()

'''
$ python analyze_power_trace.py --power_dir ../eval/20251201/logs --outdir ./plots_client
'''