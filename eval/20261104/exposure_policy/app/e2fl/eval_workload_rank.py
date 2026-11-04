import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_filters(kvs):
    out = {}
    for kv in kvs or []:
        if "=" not in kv:
            raise ValueError(f"Bad --filter '{kv}'. Use key=value (e.g., batch_size=2)")
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Merged CSV containing multiple ranks")
    ap.add_argument("--rank_col", default="lora_r", help="Rank column (e.g., lora_r)")
    ap.add_argument("--y", default="wall_time_s",
                    help="Y column: wall_time_s | time_per_step_s | time_per_token_us | max_mem_bytes")
    ap.add_argument("--filter", nargs="*", default=[],
                    help="Optional filters key=value, e.g., batch_size=2 seq_len=256 num_examples=512 local_epochs=1")
    ap.add_argument("--out", default="./y_vs_rank.png", help="Output PNG path")
    ap.add_argument("--title", default=None, help="Optional title")
    ap.add_argument("--no_errorbar", action="store_true", help="Disable std error bars")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    for col in (args.rank_col, args.y):
        if col not in df.columns:
            raise ValueError(f"Missing '{col}'. Available columns: {list(df.columns)}")

    # Apply filters (if provided)
    filters = parse_filters(args.filter)
    for k, v in filters.items():
        if k not in df.columns:
            raise ValueError(f"Filter column '{k}' not in CSV. Available: {list(df.columns)}")
        # Compare as string to be robust; then cast later
        df = df[df[k].astype(str) == str(v)]

    if df.empty:
        raise RuntimeError("No rows left after filtering. Check --filter values.")

    # Cast numeric
    df = df.copy()
    df[args.rank_col] = pd.to_numeric(df[args.rank_col], errors="coerce")
    df[args.y] = pd.to_numeric(df[args.y], errors="coerce")
    df = df.dropna(subset=[args.rank_col, args.y])

    # Aggregate mean/std per rank
    agg = df.groupby(args.rank_col)[args.y].agg(["mean", "std", "count"]).reset_index()
    agg = agg.sort_values(args.rank_col)

    plt.figure()
    xs = agg[args.rank_col].to_numpy()
    ys = agg["mean"].to_numpy()
    if args.no_errorbar:
        plt.plot(xs, ys, marker="o")
    else:
        yerr = agg["std"].fillna(0).to_numpy()
        plt.errorbar(xs, ys, yerr=yerr, fmt="-o")

    plt.xlabel(args.rank_col)
    plt.ylabel(args.y)

    if args.title:
        plt.title(args.title)
    else:
        suffix = ""
        if filters:
            suffix = " | " + ", ".join([f"{k}={v}" for k, v in filters.items()])
        plt.title(f"{args.y} vs {args.rank_col}{suffix}")

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Wrote: {args.out}")
    print(f"Used {len(df)} rows; ranks={sorted(df[args.rank_col].unique())}")


if __name__ == "__main__":
    main()