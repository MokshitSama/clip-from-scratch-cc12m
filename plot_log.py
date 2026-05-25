"""Parse train.log and plot a 2x2 PNG of loss, R@1 (both directions),
logit scale, and throughput vs. step.

Raw lines are drawn translucently; the heavy line is a 50-step rolling mean.
If multiple runs end up appended to one log (each `step 1/TOTAL` reset starts
a new run), they're plotted side by side."""
import re
import sys
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Each step line looks like:
# [00:08:30] step  5000/5000 | loss 0.4926 | scale 30.51 | lr 1.07e-10 |
#            R@1 i2t  91.8% | R@1 t2i  93.4% | 577 samples/s
LINE_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s*\|\s*"
    r"loss\s+([\d.]+)\s*\|\s*"
    r"scale\s+([\d.]+)\s*\|\s*"
    r"lr\s+([\d.eE+-]+)\s*\|\s*"
    r"R@1 i2t\s+([\d.]+)%\s*\|\s*"
    r"R@1 t2i\s+([\d.]+)%\s*\|\s*"
    r"([\d.]+)\s+samples/s"
)


def parse_log(log_path: Path) -> pd.DataFrame:
    """Parse the log into a DataFrame with a `run_id` column.

    We assign run_id by ordering: the first time `step=20` shows up is run 0,
    the second time is run 1, etc. Within a run, step is monotonic, so this
    is robust as long as runs were started near-simultaneously (which is the
    only case where they'd interleave like this).
    """
    rows = []
    text = log_path.read_text()
    for m in LINE_RE.finditer(text):
        step, total, loss, scale, lr, r1_i2t, r1_t2i, sps = m.groups()
        rows.append({
            "step":      int(step),
            "total":     int(total),
            "loss":      float(loss),
            "scale":     float(scale),
            "lr":        float(lr),
            "r1_i2t":    float(r1_i2t),
            "r1_t2i":    float(r1_t2i),
            "samples_s": float(sps),
        })

    df = pd.DataFrame(rows)

    # Assign run IDs by sequential occurrence of each step value.
    # If a step appears k times in order, the i-th occurrence goes to run i.
    df["run_id"] = df.groupby("step").cumcount()
    df = df.sort_values(["run_id", "step"]).reset_index(drop=True)
    return df


def plot(df: pd.DataFrame, out_path: Path, title: str, smooth: int = 50):
    runs = sorted(df["run_id"].unique())
    n_runs = len(runs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def panel(ax, ycol, ylabel, *, log=False, smoothing=True):
        for i, rid in enumerate(runs):
            sub = df[df["run_id"] == rid].sort_values("step")
            c = colors[i % len(colors)]
            label_raw = f"run {rid} (raw)" if n_runs > 1 else "raw"
            ax.plot(sub["step"], sub[ycol], color=c, alpha=0.25, lw=0.8,
                    label=label_raw)
            if smoothing and len(sub) >= smooth:
                roll = sub[ycol].rolling(smooth, min_periods=1).mean()
                label_sm = (f"run {rid} ({smooth}-step MA)"
                            if n_runs > 1 else f"{smooth}-step MA")
                ax.plot(sub["step"], roll, color=c, lw=2.0, label=label_sm)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    panel(axes[0, 0], "loss",      "training loss")
    # Combined R@1 panel — both directions on one axis
    ax_r = axes[0, 1]
    for i, rid in enumerate(runs):
        sub = df[df["run_id"] == rid].sort_values("step")
        c_i = colors[(2 * i) % len(colors)]
        c_t = colors[(2 * i + 1) % len(colors)]
        ax_r.plot(sub["step"], sub["r1_i2t"], color=c_i, alpha=0.20, lw=0.8)
        ax_r.plot(sub["step"], sub["r1_t2i"], color=c_t, alpha=0.20, lw=0.8)
        if len(sub) >= smooth:
            ax_r.plot(sub["step"],
                      sub["r1_i2t"].rolling(smooth, min_periods=1).mean(),
                      color=c_i, lw=2.0,
                      label=f"i2t R@1 (run {rid})" if n_runs > 1 else "i2t R@1 (MA)")
            ax_r.plot(sub["step"],
                      sub["r1_t2i"].rolling(smooth, min_periods=1).mean(),
                      color=c_t, lw=2.0,
                      label=f"t2i R@1 (run {rid})" if n_runs > 1 else "t2i R@1 (MA)")
    ax_r.set_xlabel("step")
    ax_r.set_ylabel("in-batch R@1 (%)")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=8, loc="lower right")

    panel(axes[1, 0], "scale",     "logit_scale = exp(log_scale)")
    panel(axes[1, 1], "samples_s", "throughput (samples/s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log",
                    nargs="?",
                    default="/mnt/md0/mokshit/codes/CLIP/try1/runs/layer2_run1/train.log")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: <log_dir>/train_curves.png)")
    ap.add_argument("--smooth", type=int, default=50)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out) if args.out else log_path.with_name("train_curves.png")
    title = args.title or f"Training curves — {log_path.parent.name}"

    df = parse_log(log_path)
    if df.empty:
        sys.exit(f"no step lines parsed from {log_path}")
    print(f"parsed {len(df)} rows, {df['run_id'].nunique()} run(s) "
          f"(steps {df['step'].min()}-{df['step'].max()})")

    plot(df, out_path, title, smooth=args.smooth)

    # Also print per-run summary stats so they show up in the notes.
    print("\nFinal stats per run:")
    last = df.sort_values("step").groupby("run_id").tail(10)
    summary = (last
               .groupby("run_id")
               .agg(loss=("loss", "mean"),
                    r1_i2t=("r1_i2t", "mean"),
                    r1_t2i=("r1_t2i", "mean"),
                    scale=("scale", "mean"),
                    sps=("samples_s", "mean")))
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
