"""
Visualize the rank-0 Pareto front evolving over iterations as an animated GIF.

Usage:
    python plot_pareto_gif.py --stats eval_test/nsga2_sparse_stats.pt
    python plot_pareto_gif.py --stats eval_test/nsga2_sparse_stats.pt --out eval_test/pareto.gif --fps 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=str, default="eval_test/nsga2_sparse_stats.pt",
                        help="Path to nsga2_sparse_stats.pt")
    parser.add_argument("--out", type=str, default="",
                        help="Output GIF path. Defaults to same folder as --stats.")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second in the GIF.")
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    stats_path = Path(args.stats)
    out_path = Path(args.out) if args.out else stats_path.parent / "pareto_front.gif"

    data = torch.load(stats_path, map_location="cpu", weights_only=False)
    history = data.get("history", [])

    if not history:
        print("No history found in stats file. Re-run the attack to collect per-iteration data.")
        return

    # Check if new format with rank0_objectives is available
    if "rank0_objectives" not in history[0]:
        print(
            "Stats file does not contain per-iteration Pareto front data.\n"
            "Re-run the attack to regenerate nsga2_sparse_stats.pt."
        )
        return

    # Objectives: col-0 = CE (minimized), col-1 = intersection (minimized)
    frames_ce = []
    frames_inter = []
    frames_nqry = []
    for entry in history:
        obj = np.array(entry["rank0_objectives"])  # shape (n_rank0, 2)
        frames_ce.append(obj[:, 0])                # CE
        frames_inter.append(obj[:, 1])             # intersection ratio
        frames_nqry.append(entry["nqry"])

    # Global axis limits for consistency across frames
    all_ce = np.concatenate(frames_ce)
    all_inter = np.concatenate(frames_inter)
    ce_min, ce_max = float(all_ce.min()), float(all_ce.max())
    inter_min, inter_max = float(all_inter.min()), float(all_inter.max())
    pad_ce = max((ce_max - ce_min) * 0.1, 0.1)
    pad_inter = max((inter_max - inter_min) * 0.05, 0.01)

    fig, ax = plt.subplots(figsize=(6, 5))
    scat = ax.scatter([], [], c="steelblue", s=40, edgecolors="navy", linewidths=0.5, zorder=3)
    title = ax.set_title("")
    ax.set_xlabel("Cross-Entropy (↓ minimize)")
    ax.set_ylabel("Top-k Intersection Ratio (↓ better attack)")
    ax.set_xlim(ce_min - pad_ce, ce_max + pad_ce)
    ax.set_ylim(inter_min - pad_inter, inter_max + pad_inter)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    def update(frame_idx):
        ce = frames_ce[frame_idx]
        inter = frames_inter[frame_idx]
        nqry = frames_nqry[frame_idx]

        scat.set_offsets(np.column_stack([ce, inter]))
        title.set_text(
            f"Rank-0 Pareto Front  |  iter {frame_idx + 1}/{len(history)}  |  queries={nqry}"
        )
        return scat, title

    interval_ms = int(1000 / max(1, args.fps))
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=interval_ms,
        blit=True,
    )

    writer = animation.PillowWriter(fps=args.fps)
    ani.save(str(out_path), writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved GIF → {out_path}  ({len(history)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
