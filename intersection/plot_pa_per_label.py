"""
Reproduce the "Positive Specific Agreement (PA) across all labels" bar chart:
one horizontal bar per fine label, colored by PA value, annotated with PA and
n (the number of molecules in the Goodscents/Leffingwell intersection).

Usage:
    python plot_pa_per_label.py

Inputs:
    pa_per_label_gs_lw.csv   columns: label, PA, n_valid_pairs, n_molecules

Outputs:
    pa_per_label.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------- CONFIG -------------------------
INPUT_CSV = "pa_per_label_gs_lw.csv"
OUTPUT_PNG = "pa_per_label.png"

# Labels with undefined PA (n_molecules = 0, no overlap between the two
# source datasets for that label) are excluded from the ranking and drawn
# as zero-height/gray bars at the bottom, matching the original chart.
HIGH_COLOR = "#2e8b2e"   # PA >= 0.30 (arbitrary visual split, not a statistical cutoff)
LOW_COLOR = "#c0392b"
UNDEFINED_COLOR = "#bbbbbb"
SPLIT_FOR_COLOR = 0.30
# ------------------------------------------------------------


def main():
    df = pd.read_csv(INPUT_CSV)

    defined = df[df["PA"].notna()].sort_values("PA", ascending=False)
    undefined = df[df["PA"].isna()].sort_values("label")

    plot_df = pd.concat([defined, undefined], ignore_index=True)

    colors = []
    for _, row in plot_df.iterrows():
        if pd.isna(row["PA"]):
            colors.append(UNDEFINED_COLOR)
        elif row["PA"] >= SPLIT_FOR_COLOR:
            colors.append(HIGH_COLOR)
        else:
            colors.append(LOW_COLOR)

    fig_height = max(8, 0.17 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9, fig_height))

    values = plot_df["PA"].fillna(0).values
    y_pos = range(len(plot_df))

    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df["label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Positive Specific Agreement (PA)")
    ax.set_title("Positive Specific Agreement (PA) across all labels")
    ax.set_xlim(0, max(values.max() * 1.15, 0.1))

    for y, (_, row) in zip(y_pos, plot_df.iterrows()):
        n = row["n_molecules"]
        pa_val = row["PA"]
        label_text = f"{pa_val:.2f} (n={n:.0f})" if pd.notna(pa_val) else f"undefined (n=0)"
        ax.text(row["PA"] + 0.01 if pd.notna(pa_val) else 0.005, y, label_text,
                va="center", fontsize=6, color="black")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200)
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
