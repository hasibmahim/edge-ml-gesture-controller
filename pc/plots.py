"""Make the evaluation plots and a summary table from logged runs.

Usage:  python plots.py ../results/run*.csv [--bench ../results/bench.csv]

  bytes_by_mode.png      data sent per classified window, edge vs cloud
  latency_by_mode.png    time from window ready to label, edge vs cloud
  depth_tradeoff.png     accuracy vs inference time per tree depth
                         (needs results/depth_results.csv from train.py and a
                         bench.csv with columns depth,infer_ms from bench.py)
Needs: pip install matplotlib
"""

import argparse
import csv
import glob
import os
import random
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def load_runs(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


MODES = ("edge", "hybrid", "cloud")


def bootstrap_ci(values, n=2000, seed=42):
    """95% bootstrap interval of the median (resampling windows, fixed seed)."""
    rng = random.Random(seed)
    m = len(values)
    reps = sorted(statistics.median([values[rng.randrange(m)] for _ in range(m)]) for _ in range(n))
    return reps[int(0.025 * (n - 1))], reps[int(0.975 * (n - 1))]


def median_diff_ci(a, b, n=2000, seed=42):
    """95% bootstrap interval of median(b) - median(a), for two independent runs."""
    rng = random.Random(seed)
    reps = sorted(statistics.median([b[rng.randrange(len(b))] for _ in b]) -
                  statistics.median([a[rng.randrange(len(a))] for _ in a]) for _ in range(n))
    return reps[int(0.025 * (n - 1))], reps[int(0.975 * (n - 1))]


def summarise(rows):
    out = {}
    for mode in MODES:
        r = [x for x in rows if x["mode"] == mode]
        if not r:
            continue
        ok = [x for x in r if float(x["lat_ms"]) >= 0]
        times = [float(x["time"]) for x in r]
        minutes = max((max(times) - min(times)) / 60, 1 / 60)
        total_bytes = sum(int(x["bytes"]) for x in r)
        lats = [float(x["lat_ms"]) for x in ok]
        out[mode] = {
            "windows": len(r),
            "lost": len(r) - len(ok),
            "bytes_per_window": total_bytes / len(r),
            "bytes_per_min": total_bytes / minutes,
            "lat_median_ms": statistics.median(lats) if lats else float("nan"),
            "lat_median_ci": bootstrap_ci(lats) if lats else (float("nan"), float("nan")),
            "lat_p95_ms": sorted(lats)[int(0.95 * (len(lats) - 1))] if lats else float("nan"),
            "lats": lats,
        }
    return out


def plot_modes(summary, out_dir):
    modes = [m for m in MODES if m in summary]
    colours = {"edge": "#2a7f62", "hybrid": "#b8862d", "cloud": "#3b6ea8"}
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(modes, [summary[m]["bytes_per_window"] for m in modes], color=[colours[m] for m in modes])
    ax.bar_label(bars, fmt="%.0f B")
    ax.set_ylabel("Bytes sent per classified window")
    ax.set_title("Data sent: edge vs cloud")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bytes_by_mode.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.boxplot([summary[m]["lats"] for m in modes], tick_labels=modes)
    ax.set_ylabel("Window ready → label (ms)")
    ax.set_title("Latency: edge vs cloud")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency_by_mode.png"), dpi=150)
    plt.close(fig)


def plot_depths(depth_csv, bench_csv, out_dir):
    with open(depth_csv, newline="") as f:
        acc = {int(r["depth"]): float(r["accuracy"]) for r in csv.DictReader(f)}
    with open(bench_csv, newline="") as f:
        infer = {int(r["depth"]): float(r["infer_ms"]) for r in csv.DictReader(f)}
    depths = sorted(set(acc) & set(infer))
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot([infer[d] for d in depths], [100 * acc[d] for d in depths], "o-")
    for d in depths:
        ax.annotate("depth %d" % d, (infer[d], 100 * acc[d]), textcoords="offset points", xytext=(6, -10))
    ax.set_xlabel("Inference time on the Pico (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Tree depth trade-off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "depth_tradeoff.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--bench", default=os.path.join(RESULTS, "bench.csv"))
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths = [p for pattern in args.runs for p in glob.glob(pattern)]
    if paths:
        summary = summarise(load_runs(paths))
        print("%-6s %8s %5s %12s %10s %10s %16s %8s" % ("mode", "windows", "lost", "bytes/win", "bytes/min",
                                                         "median ms", "95% CI", "p95 ms"))
        for m, s in summary.items():
            print("%-6s %8d %5d %12.0f %10.0f %10.1f %16s %8.1f" % (
                m, s["windows"], s["lost"], s["bytes_per_window"], s["bytes_per_min"], s["lat_median_ms"],
                "%.1f to %.1f" % s["lat_median_ci"], s["lat_p95_ms"]))
        if "edge" in summary and "cloud" in summary and summary["edge"]["lats"] and summary["cloud"]["lats"]:
            lo, hi = median_diff_ci(summary["edge"]["lats"], summary["cloud"]["lats"])
            print("Median latency, cloud minus edge: 95%% CI %.1f to %.1f ms" % (lo, hi))
            ratio = summary["edge"]["bytes_per_window"] / max(summary["cloud"]["bytes_per_window"], 1e-9)
            print("Bytes per window, edge / cloud: %.3f (rule in the plan: at most 0.1)" % ratio)
        plot_modes(summary, args.out)

    depth_csv = os.path.join(RESULTS, "depth_results.csv")
    if os.path.exists(depth_csv) and os.path.exists(args.bench):
        plot_depths(depth_csv, args.bench, args.out)
    print("Plots in", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
