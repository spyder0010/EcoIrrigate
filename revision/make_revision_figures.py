"""Generate publication-oriented figures only from saved prediction ledgers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parent
ARCH = ROOT / "architecture_benchmark"
BASE = ROOT / "baseline_audit"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
HORIZONS = ["1h", "6h", "12h", "24h"]
COLORS = {
    "Persistence": "#1f2937",
    "Residual Ridge": "#a16207",
    "Direct GRU": "#dc2626",
    "Residual GRU": "#2563eb",
    "Shared GRU (5 seeds)": "#0d9488",
    "Shared TCN (5 seeds)": "#7c3aed",
    "Shared BiLSTM (5 seeds)": "#be185d",
    "BiLSTM without auxiliary head (5 seeds)": "#65a30d",
}


def rmse(truth, prediction):
    return float(np.sqrt(mean_squared_error(truth, prediction)))


def main() -> None:
    architecture = pd.read_csv(ARCH / "architecture_predictions.csv", parse_dates=["origin_timestamp", "target_timestamp"])
    ridge = pd.read_csv(BASE / "residual_ridge_predictions.csv", parse_dates=["origin_timestamp", "target_timestamp"])
    common = architecture[architecture.variant.eq("shared_bilstm_residual")][["farm", "horizon", "origin_timestamp", "target_timestamp"]]
    ridge = ridge.merge(common, on=["farm", "horizon", "origin_timestamp", "target_timestamp"], validate="one_to_one")
    assert len(ridge) == len(common) == 3006 * 4
    variants = {
        "gru_direct": "Direct GRU",
        "gru_residual": "Residual GRU",
    }
    scores = {}
    errors = {}
    for horizon in HORIZONS:
        ref = architecture[(architecture.variant.eq("shared_bilstm_residual")) & (architecture.horizon.eq(horizon))]
        scores.setdefault("Persistence", []).append(rmse(ref.target_vwc_pct, ref.current_vwc_pct))
        scores.setdefault("Residual Ridge", []).append(rmse(ridge[ridge.horizon.eq(horizon)].target_vwc_pct, ridge[ridge.horizon.eq(horizon)].residual_ridge_vwc_pct))
        for code, label in variants.items():
            part = architecture[(architecture.variant.eq(code)) & (architecture.horizon.eq(horizon))]
            scores.setdefault(label, []).append(rmse(part.target_vwc_pct, part.model_vwc_pct))
        for code, label in [
            ("shared_gru_residual", "Shared GRU (5 seeds)"),
            ("shared_tcn_residual", "Shared TCN (5 seeds)"),
            ("shared_bilstm_residual", "Shared BiLSTM (5 seeds)"),
            ("bilstm_residual", "BiLSTM without auxiliary head (5 seeds)"),
        ]:
            seed_values = []
            for seed in [42, 43, 44, 45, 46]:
                if code == "bilstm_residual":
                    folder = ARCH / f"bilstm_no_aux_seed{seed}"
                else:
                    folder = ARCH if seed == 42 else ARCH / (f"seed{seed}" if code == "shared_bilstm_residual" else f"seed{seed}_others")
                data = architecture if folder == ARCH else pd.read_csv(folder / "architecture_predictions.csv")
                part = data[(data.variant.eq(code)) & (data.horizon.eq(horizon))]
                seed_values.append(rmse(part.target_vwc_pct, part.model_vwc_pct))
            scores.setdefault(label, []).append(np.mean(seed_values))
            errors.setdefault(label, []).append(np.std(seed_values, ddof=1))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    x = np.arange(len(HORIZONS))
    for label, values in scores.items():
        yerr = errors.get(label)
        ax.errorbar(x, values, yerr=yerr, label=label, marker="o", linewidth=2.0,
                    capsize=4 if yerr is not None else 0, color=COLORS[label])
    ax.set_xticks(x, ["1 hour", "6 hours", "12 hours", "24 hours"])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("RMSE (VWC percentage points)")
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8.5)
    fig.savefig(FIG / "common_origin_rmse.svg", bbox_inches="tight")
    fig.savefig(FIG / "common_origin_rmse.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "Fig2.eps", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "Fig2.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)

    no_aux = pd.read_csv(ARCH / "bilstm_no_aux_seed42" / "architecture_predictions.csv", parse_dates=["target_timestamp"])
    trace = no_aux[
        no_aux.variant.eq("bilstm_residual") &
        no_aux.horizon.eq("6h") &
        no_aux.farm.eq("Kolkata_Farm_1")
    ].sort_values("target_timestamp").iloc[:96 * 5:4]
    fig, ax = plt.subplots(figsize=(9.2, 4.5), constrained_layout=True)
    ax.plot(trace.target_timestamp, trace.target_vwc_pct, color=COLORS["Persistence"], lw=2.0, label="Supplied VWC target")
    ax.plot(trace.target_timestamp, trace.current_vwc_pct, color="#9ca3af", ls="--", label="Persistence (6 h)")
    ax.plot(trace.target_timestamp, trace.model_vwc_pct, color=COLORS["BiLSTM without auxiliary head (5 seeds)"], lw=1.6, label="Residual BiLSTM (seed 42)")
    ax.set_xlabel("Target timestamp in held-out test period")
    ax.set_ylabel("Supplied VWC (%)")
    ax.grid(color="#e5e7eb")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8.5)
    fig.autofmt_xdate()
    fig.savefig(FIG / "six_hour_trace.svg", bbox_inches="tight")
    fig.savefig(FIG / "six_hour_trace.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "Fig3.eps", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "Fig3.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)

    print("Wrote", *[p.name for p in FIG.iterdir()], sep="\n")


if __name__ == "__main__":
    main()
