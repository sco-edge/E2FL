"""Local workload micro-benchmark for E2FL (no FL).

Goal
----
Measure how local fine-tuning time changes as we vary workload knobs on a single device
(e.g., Jetson Orin NX). Generates a CSV (and optional plot) for workload→time curves.

We default to a *synthetic* causal-LM dataset to reduce variance from:
- tokenization/preprocessing
- disk I/O
- heavy collators

Example
-------
python -m e2fl.eval_workload \
  --model "hf:google/gemma-3-270m" \
  --device cuda \
  --sweep seq_len --values 64 128 256 384 512 \
  --batch_size 2 --num_examples 1024 --local_epochs 1 \
  --repeats 20 --warmup 2 --plot
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# Try to reuse your project model factory when possible
try:
    from e2fl.models import get_model, is_llm_model_name
except Exception:
    get_model = None
    is_llm_model_name = lambda _: True  # type: ignore


# -----------------------------
# Synthetic causal-LM dataset
# -----------------------------
class SyntheticCausalLMDataset(Dataset):
    """Returns dict batches compatible with common HF causal-LM training loops.

    Each sample is:
      input_ids: (seq_len,)
      attention_mask: (seq_len,)
      labels: (seq_len,)
    """

    def __init__(self, num_examples: int, seq_len: int, vocab_size: int, seed: int = 0):
        self.num_examples = int(num_examples)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.base_seed = int(seed)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Deterministic per idx (reproducible without storing data)
        g = torch.Generator()
        g.manual_seed(self.base_seed + int(idx))
        input_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len,),
            dtype=torch.long,
            generator=g,
        )
        attention_mask = torch.ones((self.seq_len,), dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _default_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


# -----------------------------
# Benchmark core
# -----------------------------
@dataclass
class TrialResult:
    timestamp: str
    model: str
    device: str
    amp: str
    lora_r: str
    batch_size: int
    seq_len: int
    num_examples: int
    local_epochs: int
    steps: int
    tokens: int
    wall_time_s: float
    time_per_step_s: float
    time_per_token_us: float
    max_mem_bytes: int


def _get_vocab_size(model: torch.nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "vocab_size"):
        try:
            vs = int(getattr(cfg, "vocab_size"))
            if vs > 0:
                return vs
        except Exception:
            pass
    return 32000


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_mem(device: torch.device) -> int:
    if device.type == "cuda":
        try:
            return int(torch.cuda.max_memory_allocated(device))
        except Exception:
            return 0
    return 0


def run_one_trial(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    local_epochs: int,
    lr: float,
    amp: str,
) -> Tuple[float, int]:
    """Run one local fine-tuning trial. Returns (wall_time_s, max_mem_bytes)."""
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    use_amp = amp.lower() in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.float16 if amp.lower() == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp.lower() == "fp16"))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    _sync(device)
    t0 = time.perf_counter()

    for _ in range(int(local_epochs)):
        for batch in loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    out = model(**batch)
                    loss = out.loss if hasattr(out, "loss") else out[0]
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                out = model(**batch)
                loss = out.loss if hasattr(out, "loss") else out[0]
                loss.backward()
                optimizer.step()

    _sync(device)
    t1 = time.perf_counter()

    return (t1 - t0), _peak_mem(device)


def build_model(model_name: str, lora_r: Optional[int]) -> torch.nn.Module:
    # Allow overriding LoRA rank via env var (matches your client_app_llm.py style)
    if lora_r is not None:
        os.environ["LORA_R"] = str(int(lora_r))

    if get_model is None:
        raise RuntimeError("e2fl.models.get_model import failed. Run this inside the EEFL repo.")

    num_classes = 0 if is_llm_model_name(model_name) else 10
    try:
        return get_model(model_name, num_classes)
    except OSError as e:
        hint = (
            "Failed to load model. If you're using a Hugging Face model, pass the full identifier "
            "used elsewhere in this repo (e.g., --model 'hf:google/gemma-3-270m' or 'google/gemma-3-270m'). "
            "Some models are gated/private; in that case run `hf auth login` (or set an access token) and ensure you have access."
        )
        raise RuntimeError(f"{hint}\nOriginal error: {e}") from e


def ensure_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda:0")
    if device_str == "cpu":
        return torch.device("cpu")
    return torch.device(device_str)


def write_csv(path: str, rows: List[TrialResult]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def plot_summary(rows: List[TrialResult], sweep: str, out_png: str) -> None:
    import statistics
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })

    groups: Dict[int, List[float]] = {}
    for r in rows:
        v = int(getattr(r, sweep))
        groups.setdefault(v, []).append(float(r.wall_time_s))

    xs = sorted(groups.keys())
    means = [statistics.mean(groups[x]) for x in xs]
    stds = [statistics.pstdev(groups[x]) if len(groups[x]) > 1 else 0.0 for x in xs]

    plt.figure()
    plt.errorbar(xs, means, yerr=stds, fmt="-", color="black", linewidth=2.0, capsize=3)
    plt.xlabel(sweep)
    plt.ylabel("time (s)")
    plt.title(f"Local fine-tuning time vs {sweep}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)


def snapshot_state_dict_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, required=True, help="Model identifier passed to e2fl.models.get_model (e.g., 'hf:google/gemma-3-270m')")
    p.add_argument("--device", type=str, default="cuda", help="cuda|cpu|cuda:0")
    p.add_argument("--amp", type=str, default="bf16", choices=["none", "fp16", "bf16"], help="AMP mode (CUDA only)")
    p.add_argument("--lora_r", type=int, default=None, help="Override LoRA rank via env LORA_R")

    # Baseline knobs
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--num_examples", type=int, default=1024)
    p.add_argument("--local_epochs", type=int, default=1)

    # Sweep
    p.add_argument("--sweep", type=str, default="seq_len",
                   choices=["batch_size", "seq_len", "num_examples", "local_epochs"])
    p.add_argument("--values", type=int, nargs="+", required=True, help="Sweep values")

    # Repeats/warmup
    p.add_argument("--warmup", type=int, default=2, help="Warmup trials per sweep value")
    p.add_argument("--repeats", type=int, default=10, help="Measured trials per sweep value")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate used in benchmark")

    # Output
    p.add_argument("--out_dir", type=str, default="./eval/workload_bench")
    p.add_argument("--tag", type=str, default="", help="Optional tag added to filenames")
    p.add_argument("--plot", action="store_true", help="Also write a PNG plot")

    # Reset model between trials (default True)
    p.add_argument("--no_reset_model", action="store_true", help="Disable resetting model weights between trials")

    args = p.parse_args()

    device = ensure_device(args.device)

    # Repro
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    # Build model once (keep consistent across trials)
    model = build_model(args.model, args.lora_r)
    model.to(device)

    init_state = snapshot_state_dict_cpu(model)

    reset_model = not args.no_reset_model

    vocab_size = _get_vocab_size(model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = ("_" + args.tag) if args.tag else ""
    csv_path = os.path.join(args.out_dir, f"bench_{args.model}_{device.type}_{args.sweep}{tag}_{timestamp}.csv")
    png_path = os.path.join(args.out_dir, f"bench_{args.model}_{device.type}_{args.sweep}{tag}_{timestamp}.png")

    rows: List[TrialResult] = []

    base_knobs = {
        "batch_size": int(args.batch_size),
        "seq_len": int(args.seq_len),
        "num_examples": int(args.num_examples),
        "local_epochs": int(args.local_epochs),
    }

    for v in args.values:
        knobs = dict(base_knobs)
        knobs[args.sweep] = int(v)

        ds = SyntheticCausalLMDataset(
            num_examples=knobs["num_examples"],
            seq_len=knobs["seq_len"],
            vocab_size=vocab_size,
            seed=int(args.seed),
        )
        loader = DataLoader(
            ds,
            batch_size=knobs["batch_size"],
            shuffle=True,
            drop_last=False,  # process the full dataset (fixed data amount)
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            collate_fn=_default_collate,
        )

        steps_per_epoch = max(1, math.ceil(len(ds) / knobs["batch_size"]))
        steps = steps_per_epoch * knobs["local_epochs"]
        tokens = len(ds) * knobs["seq_len"] * knobs["local_epochs"]

        # Warmup (not recorded)
        for _ in range(int(args.warmup)):
            if reset_model:
                model.load_state_dict(init_state, strict=True)
                if device.type == "cuda":
                    _sync(device)
            _ = run_one_trial(model, loader, device, knobs["local_epochs"], float(args.lr), str(args.amp))

        # Measured repeats
        vals = []
        for _ in range(int(args.repeats)):
            if reset_model:
                model.load_state_dict(init_state, strict=True)
                if device.type == "cuda":
                    _sync(device)
            wall_time, max_mem = run_one_trial(model, loader, device, knobs["local_epochs"], float(args.lr), str(args.amp))
            vals.append(wall_time)

            rows.append(
                TrialResult(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    model=str(args.model),
                    device=str(device),
                    amp=str(args.amp),
                    lora_r=str(args.lora_r) if args.lora_r is not None else "",
                    batch_size=knobs["batch_size"],
                    seq_len=knobs["seq_len"],
                    num_examples=knobs["num_examples"],
                    local_epochs=knobs["local_epochs"],
                    steps=int(steps),
                    tokens=int(tokens),
                    wall_time_s=float(wall_time),
                    time_per_step_s=float(wall_time / max(1, steps)),
                    time_per_token_us=float((wall_time * 1e6) / max(1, tokens)),
                    max_mem_bytes=int(max_mem),
                )
            )

        mean = sum(vals) / len(vals)
        print(f"[{args.sweep}={v}] collected {len(vals)} trials, mean={mean:.3f}s")

    if not rows:
        raise RuntimeError("No results collected. Did you set --values?")

    write_csv(csv_path, rows)
    print(f"Wrote CSV: {csv_path}")

    if args.plot:
        plot_summary(rows, args.sweep, png_path)
        print(f"Wrote plot: {png_path}")


if __name__ == "__main__":
    main()