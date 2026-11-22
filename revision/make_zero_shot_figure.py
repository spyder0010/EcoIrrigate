"""Show all eligible unseen-station RMSE differences without selecting a win."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "external_noaa" / "zero_shot" / "zero_shot_2026.json"


def main() -> None:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    rng = np.random.default_rng(20261114)
    plt.rcParams.update({"ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    colors = ("#475569", "#0f766e", "#1d4ed8", "#7c3aed")
    for i, horizon in enumerate((1, 6, 12, 24), start=1):
        rows = [r for r in report["station_results"] if r["horizon_hours"] == horizon and r["eligible"]]
        pooled = next(r for r in report["pooled_results"] if r["horizon_hours"] == horizon)
        values = np.array([r["persistence_rmse"] - r["ridge_rmse"] for r in rows])
        jitter = rng.uniform(-0.17, 0.17, len(values))
        ax.scatter(i + jitter, values, s=22, color=colors[i - 1], linewidths=0, zorder=2)
        q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
        ax.plot([i, i], [q1, q3], color="#111827", lw=4, solid_capstyle="round", zorder=3)
        ax.plot([i - 0.13, i + 0.13], [median, median], color="#111827", lw=2, zorder=4)
        pooled_skill = pooled["persistence_rmse"] - pooled["ridge_rmse"]
        ax.scatter([i], [pooled_skill], marker="D", s=52, facecolor="#f59e0b", edgecolor="#111827", zorder=5)
        ax.text(i, 0.27, f"{pooled['stations_ridge_better_rmse']}/{pooled['n_stations']} sites", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="#991b1b", lw=1, ls="--")
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(-0.55, 0.32)
    ax.set_xticks((1, 2, 3, 4), ("1 h", "6 h", "12 h", "24 h"))
    ax.set_ylabel("Persistence RMSE − Ridge RMSE (VWC points)")
    ax.set_xlabel("Forecast horizon")
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    ax.scatter([], [], marker="D", s=52, facecolor="#f59e0b", edgecolor="#111827", label="Pooled difference")
    ax.plot([], [], color="#111827", lw=4, label="Station median and middle half")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "noaa_zero_shot_station_distribution.png", dpi=260, bbox_inches="tight")
    fig.savefig(out / "noaa_zero_shot_station_distribution.svg", bbox_inches="tight")
    fig.savefig(out / "Fig5.eps", dpi=600, bbox_inches="tight")
    fig.savefig(out / "Fig5.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    print("Saved", out / "noaa_zero_shot_station_distribution.png")


if __name__ == "__main__":
    main()
