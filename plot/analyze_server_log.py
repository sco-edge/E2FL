import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# =========================================================
# 0) CID → device 매핑 (반드시 사용자가 수동으로 작성)
# =========================================================
CID_TO_DEVICE = {
    1.64611E+19: "Jetson Orin NX",
    1.74163E+19: "RPi 5",
}


# =========================================================
# 1) CSV 로드 + Jetson 보정
# =========================================================
def load_server_csv(csv_path: str, needs_jetson_scale: bool) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=r"\s*,\s*",
        engine="python",
    )

    # timestamp 문자열 → datetime 컬럼 생성 (옵션)
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(
            df["timestamp"],
            format="%Y-%m-%d_%H-%M-%S",
            errors="coerce"
        )

    # -------------------------------
    # device 컬럼 생성 (매우 중요)
    # -------------------------------
    if "cid" in df.columns:
        df["device"] = df["cid"].apply(
            lambda x: CID_TO_DEVICE[x] if x in CID_TO_DEVICE else f"cid_{x}"
        )
    else:
        df["device"] = "unknown"

    #df["device"] = df["device"].astype(str).str.lower()

    # -------------------------------
    # phase를 카테고리로 변환
    # -------------------------------
    if "phase" in df.columns:
        df["phase"] = df["phase"].astype("category")

    # -------------------------------
    # ⭐ Jetson 전력 보정 (/1000)
    # device 이름은 CID_TO_DEVICE 에 의해 "Jetson Orin NX" 로 매핑됨
    if needs_jetson_scale and "energy" in df.columns:
        mask_jetson = df["device"].astype(str).str.lower().str.contains("jetson")
        df.loc[mask_jetson, "energy"] = df.loc[mask_jetson, "energy"] / 1000.0

    # -------------------------------
    # 출력 확인 (optional)
    # -------------------------------
    print("\n[INFO] Loaded DataFrame (head):")
    print(df.head())
    print(f"\nShape: {df.shape}")

    return df


# =========================================================
# 2) 라운드/phase 집계
# =========================================================
def aggregate_by_round_phase(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["round", "phase"])
        .agg(
            num_examples=("num_examples", "sum"),
            loss=("loss", "mean"),
            accuracy=("accuracy", "mean"),
            energy=("energy", "sum"),
            comm_time=("comm_time", "sum"),
            phase_sent=("phase_sent", "sum"),
            phase_recv=("phase_recv", "sum"),
        )
        .reset_index()
    )
    return agg


def weighted_eval_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    df_eval = df[df["phase"] == "evaluate"]

    if df_eval.empty or df_eval["accuracy"].isna().all():
        return pd.DataFrame(columns=["round", "accuracy_weighted"])

    def _wavg(group):
        return (group["accuracy"] * group["num_examples"]).sum() / group["num_examples"].sum()

    return (
        df_eval.groupby("round")
        .apply(_wavg)
        .reset_index(name="accuracy_weighted")
    )


# =========================================================
# 3) 기본 그래프들
# =========================================================
def plot_train_loss_over_round(agg_round_phase, outdir):
    df_train = agg_round_phase[agg_round_phase["phase"] == "train"]
    if df_train.empty:
        return

    plt.figure(figsize=(8, 5))
    sns.lineplot(x="round", y="loss", data=df_train, marker="o")
    #plt.title("Train Loss per Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "train_loss_per_round.png")
    plt.close()


def plot_eval_accuracy_over_round(weighted_acc, outdir):
    if weighted_acc.empty:
        return

    plt.figure(figsize=(8, 5))
    sns.lineplot(x="round", y="accuracy_weighted", data=weighted_acc, marker="o")
    #plt.title("Eval Accuracy per Round (weighted)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "eval_accuracy_per_round.png")
    plt.close()


def plot_energy_over_round(agg_round_phase, outdir):
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        x="round",
        y="energy",
        hue="phase",
        data=agg_round_phase,
        marker="o",
    )
    #plt.title("Energy per Round (Train vs Evaluate)")
    plt.xlabel("Round")
    plt.ylabel("Mean Power (W)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "energy_per_round_phase.png")
    plt.close()


def plot_commtime_over_round(agg_round_phase, outdir):
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        x="round",
        y="comm_time",
        hue="phase",
        data=agg_round_phase,
        marker="o"
    )
    plt.yscale("log")
    #plt.title("Communication Time per Round (log scale)")
    plt.xlabel("Round")
    plt.ylabel("Comm Time (log scale)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "commtime_per_round_phase.png")
    plt.close()


# =========================================================
# 4) 디바이스 단위 비교
# =========================================================
def plot_energy_per_round_per_device(df, outdir):
    agg = (
        df.groupby(["round", "device"])
        .agg(energy=("energy", "sum"))
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        x="round",
        y="energy",
        hue="device",
        data=agg,
        marker="o"
    )
    #plt.title("Energy per Round by Device")
    plt.xlabel("Round")
    plt.ylabel("Mean Power (W)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "energy_per_round_device.png")
    plt.close()


def plot_commtime_per_round_per_device(df, outdir):
    agg = (
        df.groupby(["round", "device"])
        .agg(comm_time=("comm_time", "sum"))
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        x="round",
        y="comm_time",
        hue="device",
        data=agg,
        marker="o"
    )
    plt.yscale("log")
    #plt.title("Comm Time per Round by Device (log scale)")
    plt.xlabel("Round")
    plt.ylabel("Comm Time (log scale)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "commtime_per_round_device.png")
    plt.close()


# =========================================================
# 5) main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="server CSV path")
    parser.add_argument("--outdir", type=str, default="plots_server")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 날짜 기반 Jetson energy scaling 여부 결정
    folder_str = str(csv_path.parent)
    needs_jetson_scale = ("2025-12-01" in folder_str) or ("20251201" in folder_str)

    print(f"[INFO] Jetson energy scaling applied? → {needs_jetson_scale}")

    # CSV 로딩 + 보정
    df = load_server_csv(str(csv_path), needs_jetson_scale)

    # 집계
    agg_round_phase = aggregate_by_round_phase(df)
    weighted_acc = weighted_eval_accuracy(df)

    # 그래프
    plot_train_loss_over_round(agg_round_phase, outdir)
    plot_eval_accuracy_over_round(weighted_acc, outdir)
    plot_energy_over_round(agg_round_phase, outdir)
    plot_commtime_over_round(agg_round_phase, outdir)

    plot_energy_per_round_per_device(df, outdir)
    plot_commtime_per_round_per_device(df, outdir)

    print(f"[INFO] Saved plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

'''
$ python analyze_server_log.py ../eval/20251201/logs/fedavg_2025-12-01_16-39-58.csv --outdir plots_server/20251201/
'''