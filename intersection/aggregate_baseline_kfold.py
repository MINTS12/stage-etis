"""
Aggregate baseline_fine138_kfold_all_results.csv across the 5 folds and,
for each (Model, Label), select the best Config based on mean PR_AUC.

Usage:
    python aggregate_baseline_kfold.py

Edit INPUT_CSV / OUTPUT_DIR below if paths differ.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------- CONFIG -------------------------
INPUT_CSV = "baseline_fine138_kfold_all_results.csv"
OUTPUT_DIR = "kfold_results"

N_FOLDS_EXPECTED = 5

# Expected configs per model, in the order they were run.
# Used only as a sanity check against what's actually in the CSV.
EXPECTED_CONFIGS = {
    "LR": [
        "LR_C=0.01", "LR_C=0.1", "LR_C=1", "LR_C=10", "LR_C=100",
    ],
    "RF": [
        "RF_n_estimators=100_max_features=sqrt",
        "RF_n_estimators=300_max_features=sqrt",
        "RF_n_estimators=100_max_features=log2",
        "RF_n_estimators=300_max_features=log2",
    ],
    "XGB": [
        "XGB_n_estimators=100_max_depth=6",
        "XGB_n_estimators=100_max_depth=3",
        "XGB_n_estimators=300_max_depth=6",
        "XGB_n_estimators=300_max_depth=3",
    ],
    "SVM": [
        "SVM_C=0.1", "SVM_C=1", "SVM_C=10",
    ],
    "KNN": [
        "KNN_n_neighbors=5_metric=euclidean",
        "KNN_n_neighbors=5_metric=cosine",
        "KNN_n_neighbors=11_metric=euclidean",
        "KNN_n_neighbors=11_metric=cosine",
        "KNN_n_neighbors=21_metric=euclidean",
        "KNN_n_neighbors=21_metric=cosine",
    ],
}

METRIC_COLS = [
    "Bal.Acc", "MCC", "F1", "ROC_AUC", "PR_AUC",
    "Precision", "Sensitivity", "Specificity",
]
# ------------------------------------------------------------


def sanity_check_configs(df: pd.DataFrame) -> None:
    """Warn if the configs found in the CSV don't match EXPECTED_CONFIGS."""
    for model, expected in EXPECTED_CONFIGS.items():
        found = sorted(df.loc[df["Model"] == model, "Config"].unique())
        if sorted(expected) != found:
            print(f"[WARNING] Config mismatch for model '{model}':")
            print(f"          expected: {sorted(expected)}")
            print(f"          found:    {found}")


def main():
    df = pd.read_csv(INPUT_CSV)

    # --- basic sanity checks ---
    n_folds = df["Fold"].nunique()
    if n_folds != N_FOLDS_EXPECTED:
        print(f"[WARNING] Found {n_folds} unique folds, expected {N_FOLDS_EXPECTED}.")
    sanity_check_configs(df)

    # --- aggregate mean/std across folds for every (Model, Config, Label) ---
    group_cols = ["Model", "Config", "Label"]
    agg_dict = {col: ["mean", "std", "count"] for col in METRIC_COLS}
    summary = df.groupby(group_cols).agg(agg_dict)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    # flag any (Model, Config, Label) combos that don't have exactly N_FOLDS_EXPECTED rows
    incomplete = summary[summary["PR_AUC_count"] != N_FOLDS_EXPECTED]
    if not incomplete.empty:
        print(f"[WARNING] {len(incomplete)} (Model, Config, Label) combos do not have "
              f"{N_FOLDS_EXPECTED} fold results. Check for missing/failed runs.")

    # --- for each (Model, Label), pick the Config with the highest mean PR_AUC ---
    best_idx = summary.groupby(["Model", "Label"])["PR_AUC_mean"].idxmax()
    best_config = summary.loc[best_idx].sort_values(["Model", "Label"]).reset_index(drop=True)

    # --- save outputs ---
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "baseline_fine138_kfold_summary_all_configs.csv"
    best_path = out_dir / "baseline_fine138_kfold_best_config_per_label.csv"

    summary.to_csv(summary_path, index=False)
    best_config.to_csv(best_path, index=False)

    print(f"\nSaved full mean/std summary ({len(summary)} rows) to: {summary_path}")
    print(f"Saved best-config-per-label ({len(best_config)} rows) to: {best_path}")

    # --- quick console preview: macro-average of best PR_AUC per model ---
    macro_summary = (
        best_config.groupby("Model")["PR_AUC_mean"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\nMacro-mean of (per-label best) PR_AUC, by model:")
    print(macro_summary.to_string())


if __name__ == "__main__":
    main()
