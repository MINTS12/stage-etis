"""
Scatter plot of Positive Specific Agreement (PA) vs. baseline PR-AUC,
one point per fine label, bubble size proportional to n_PA (the number of
molecules in the Goodscents/Leffingwell intersection for that label).
Points are colored by whether n_PA is above or below the median, since PA
estimates from very few molecules are unreliable.

Usage:
    python plot_pa_vs_prauc_scatter.py

Inputs:
    pa_per_label_gs_lw.csv               columns: label, PA, n_valid_pairs, n_molecules
    best_model_per_label.csv             columns: Label, Model, Config, PR_AUC_mean, ...

Outputs:
    pa_vs_prauc_merged.csv   (merged label-level table, for reuse)
    pa_vs_prauc_scatter.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------- CONFIG -------------------------
PA_CSV = "pa_per_label_gs_lw.csv"
BEST_MODEL_CSV = "best_model_per_label.csv"
MERGED_OUT_CSV = "pa_vs_prauc_merged.csv"
SCATTER_OUT_PNG = "pa_vs_prauc_scatter.png"
# ------------------------------------------------------------


def main():
    pa = pd.read_csv(PA_CSV)
    best = pd.read_csv(BEST_MODEL_CSV)

    # Best-model file uses "fine_<label>" naming; PA file uses bare label names.
    best = best.copy()
    best["label_clean"] = best["Label"].str.replace("fine_", "", regex=False).str.strip()
    pa = pa.copy()
    pa["label_clean"] = pa["label"].str.strip()
    pa = pa.rename(columns={"n_molecules": "n_PA"})

    df = best.merge(pa, on="label_clean", how="inner")
    df.to_csv(MERGED_OUT_CSV, index=False)
    print(f"Saved merged table: {MERGED_OUT_CSV}  ({len(df)} labels)")

    # Labels with undefined PA (n_PA = 0) cannot be plotted or correlated.
    valid = df.dropna(subset=["PA"]).copy()
    n_dropped = len(df) - len(valid)
    if n_dropped:
        print(f"Dropped {n_dropped} labels with undefined PA (zero GS/LW overlap).")

    cutoff = valid["n_PA"].median()
    valid["group"] = np.where(valid["n_PA"] >= cutoff, "high", "low")

    # --- correlation summary printed to console ---
    rho_all, p_all = stats.spearmanr(valid["PA"], valid["PR_AUC_mean"])
    high = valid[valid["group"] == "high"]
    rho_high, p_high = stats.spearmanr(high["PA"], high["PR_AUC_mean"])
    print(f"\nSpearman PA vs PR_AUC, all {len(valid)} labels: rho={rho_all:.3f}, p={p_all:.2e}")
    print(f"Spearman PA vs PR_AUC, n_PA >= {cutoff:.1f} ({len(high)} labels): "
          f"rho={rho_high:.3f}, p={p_high:.2e}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    for group, color, label in [
        ("low", "#bbbbbb", f"n_PA < {cutoff:.1f}"),
        ("high", "#2a78d6", f"n_PA >= {cutoff:.1f}"),
    ]:
        sub = valid[valid["group"] == group]
        sizes = np.clip(np.sqrt(sub["n_PA"]) * 12, 15, None)
        ax.scatter(sub["PA"], sub["PR_AUC_mean"], s=sizes, alpha=0.6,
                   color=color, edgecolor="white", linewidth=0.5, label=label)

    ax.set_xlabel("Positive Specific Agreement (PA)")
    ax.set_ylabel("Baseline PR-AUC (best config per label)")
    ax.set_title("PA vs. baseline PR-AUC across fine labels")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(SCATTER_OUT_PNG, dpi=200)
    print(f"\nSaved: {SCATTER_OUT_PNG}")


if __name__ == "__main__":
    main()
