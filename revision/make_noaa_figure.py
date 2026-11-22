"""Draw the external 2026 station-level skill figure from saved scores."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "external_noaa" / "site_blend_2026_expanded_heldout.json"
STATIONS = (
    ("AZ_Tucson_11_W", "Arizona · Tucson"),
    ("CA_Merced_23_WSW", "California · Merced"),
    ("FL_Sebring_23_SSE", "Florida · Sebring"),
    ("IA_Des_Moines_17_E", "Iowa · Des Moines"),
    ("IL_Champaign_9_SW", "Illinois · Champaign"),
    ("SC_Blackville_3_W", "South Carolina · Blackville"),
    ("TX_Austin_33_NW", "Texas · Austin"),
)
HORIZONS = (1, 6, 12, 24)


def main() -> None:
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))["results"]
    lookup = {(r["station"], r["horizon_hours"]): r for r in rows}
    skill = np.array([
        [100 * (lookup[(s, h)]["persistence_rmse"] - lookup[(s, h)]["blend_rmse"]) / lookup[(s, h)]["persistence_rmse"]
         for h in HORIZONS]
        for s, _ in STATIONS
    ])
    limit = float(np.ceil(np.abs(skill).max() / 5) * 5)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.5, 5.9), constrained_layout=True)
    # QuadMesh keeps the 28 heatmap cells as vector paths in the EPS output.
    # imshow embedded a full-size 600 dpi raster and inflated Fig4.eps to 67 MB.
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = plt.get_cmap("RdBu")
    image = ax.pcolormesh(np.arange(len(HORIZONS) + 1) - 0.5,
                          np.arange(len(STATIONS) + 1) - 0.5,
                          skill, cmap=cmap, norm=norm,
                          shading="flat")
    ax.set_xlim(-0.5, len(HORIZONS) - 0.5)
    ax.set_ylim(len(STATIONS) - 0.5, -0.5)
    ax.set_xticks(range(len(HORIZONS)), [f"{h} h" for h in HORIZONS])
    ax.set_yticks(range(len(STATIONS)), [label for _, label in STATIONS])
    ax.set_xlabel("Forecast horizon")
    for i in range(skill.shape[0]):
        for j in range(skill.shape[1]):
            rgb = cmap(norm(skill[i, j]))[:3]
            linear = [channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
                      for channel in rgb]
            luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
            white_contrast = 1.05 / (luminance + 0.05)
            black_contrast = (luminance + 0.05) / 0.05
            ink = "white" if white_contrast > black_contrast else "black"
            ax.text(j, i, f"{skill[i, j]:+.1f}%", ha="center", va="center", color=ink, fontsize=9)
    # A bounded number of colorbar bands prevents Matplotlib from rasterizing
    # its gradient in PostScript; all five submission EPS figures stay vector.
    colorbar = fig.colorbar(image, ax=ax, shrink=0.87, pad=0.03,
                            boundaries=np.linspace(-limit, limit, 31),
                            ticks=np.linspace(-limit, limit, 7))
    colorbar.set_label("RMSE reduction relative to persistence (%)")
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "noaa_2026_site_skill.png", dpi=250, bbox_inches="tight")
    fig.savefig(out / "noaa_2026_site_skill.svg", bbox_inches="tight")
    fig.savefig(out / "Fig4.eps", dpi=600, bbox_inches="tight")
    fig.savefig(out / "Fig4.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"Saved {out / 'noaa_2026_site_skill.png'}")


if __name__ == "__main__":
    main()
