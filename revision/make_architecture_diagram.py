"""Draw the implemented shared-encoder model and its forecast-time boundary."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from forecast_architecture_benchmark import FEATURES, HORIZON_NAMES, WINDOW


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)


def box(ax, xy, width, height, title, detail, face):
    x, y = xy
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.08,rounding_size=0.12",
                           linewidth=1.3, edgecolor="#334155", facecolor=face)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.65, title, ha="center", va="center", fontsize=11, weight="bold")
    ax.text(x + width / 2, y + height * 0.28, detail, ha="center", va="center", fontsize=8.8)


def arrow(ax, start, end, color="#475569"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.6, color=color, connectionstyle="arc3,rad=0"))


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(13.2, 5.8), constrained_layout=True)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0.6, 6.0)
    ax.axis("off")

    box(ax, (0.35, 4.65), 2.55, 1.2, "History", f"t−96 … t−1  |  {WINDOW} × {len(FEATURES)}", "#e0f2fe")
    box(ax, (3.65, 4.65), 2.55, 1.2, "Shared encoder", "GRU / left-padded TCN / BiLSTM", "#dbeafe")
    box(ax, (7.1, 4.65), 2.15, 1.2, "Context", "64 learned values", "#e0e7ff")
    arrow(ax, (2.9, 5.25), (3.65, 5.25))
    arrow(ax, (6.2, 5.25), (7.1, 5.25))

    box(ax, (0.35, 2.7), 2.55, 1.2, "Current sensor", "ADC + voltage at t  |  2", "#fef3c7")
    box(ax, (0.35, 0.75), 2.55, 1.2, "Current state", "VWC + environment at t  |  1 + 6", "#dcfce7")
    box(ax, (3.65, 2.7), 2.55, 1.2, "Site ID", "embedding  |  4", "#f1f5f9")

    box(ax, (7.1, 2.7), 2.15, 1.2, "Auxiliary head", "64 GELU → 1", "#fef3c7")
    box(ax, (10.35, 2.7), 2.45, 1.2, "Reconstruct VWC(t)", "sensor-derived target", "#fffbeb")
    box(ax, (7.1, 0.75), 2.15, 1.2, "Forecast head", "64 GELU → 4 deltas", "#dcfce7")
    box(ax, (10.35, 0.75), 2.45, 1.2, "Forecast VWC", "t + 1, 6, 12, 24 h", "#f0fdf4")

    arrow(ax, (8.18, 4.65), (8.18, 3.9), "#1d4ed8")
    ax.plot([7.1, 6.55, 6.55], [5.25, 5.25, 1.7], color="#1d4ed8", lw=1.6)
    arrow(ax, (6.55, 1.7), (7.1, 1.7), "#1d4ed8")
    ax.plot([2.9, 3.15, 6.85, 6.85], [3.3, 2.45, 2.45, 3.0], color="#b45309", lw=1.6)
    arrow(ax, (6.85, 3.0), (7.1, 3.0), "#b45309")
    arrow(ax, (6.2, 3.5), (7.1, 3.5))
    arrow(ax, (2.9, 1.35), (7.1, 1.35))
    arrow(ax, (5.0, 2.7), (5.0, 1.15))
    arrow(ax, (5.0, 1.15), (7.1, 1.15))
    arrow(ax, (9.25, 3.3), (10.35, 3.3))
    arrow(ax, (9.25, 1.35), (10.35, 1.35))
    ax.text(9.8, 1.6, "+ VWC(t)", ha="center", fontsize=9, color="#166534")
    for suffix, kwargs in (("svg", {}), ("png", {"dpi": 300})):
        fig.savefig(OUT / f"shared_architecture.{suffix}", bbox_inches="tight", **kwargs)
    fig.savefig(OUT / "Fig1.eps", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "Fig1.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    print("Wrote shared_architecture.svg and .png")


if __name__ == "__main__":
    main()
