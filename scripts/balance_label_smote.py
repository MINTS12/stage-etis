"""
Simplified KNN SMOTE.
Requires you to manually specify the target label.

Usage:
    python balance_label_smote.py --input jadbio_spicy.csv --target spicy
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

def smote_knn(X, n_synthetic, binary_mask, k=5, seed=42):
    """
    KNN SMOTE with binary-aware interpolation.
    """
    rng = np.random.default_rng(seed)
    n   = X.shape[0]
    k   = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
    nn.fit(X)
    _, idx = nn.kneighbors(X)

    out = np.zeros((n_synthetic, X.shape[1]))
    for i in range(n_synthetic):
        bi = rng.integers(0, n)
        ni = idx[bi, rng.integers(1, k + 1)]
        a  = rng.uniform(0, 1)
        s  = X[bi] + a * (X[ni] - X[bi])
        # round binary features, leave continuous as-is
        s[binary_mask] = np.round(s[binary_mask]).clip(0, 1)
        out[i] = s
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Input CSV file path")
    parser.add_argument("--target", required=True, help="Target column name to balance (e.g., spicy)")
    parser.add_argument("--ratio",  type=float, default=0.5, help="Target positive ratio (default 0.5)")
    parser.add_argument("--k",      type=int,   default=5,   help="KNN neighbors (default 5)")
    parser.add_argument("--sep",    default=";",             help="CSV separator (default ';')")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {args.input}")
        return

    print(f"\nLoading {input_path.name}...")
    df = pd.read_csv(input_path, sep=args.sep)
    target = args.target

    if target not in df.columns:
        print(f"ERROR: Target column '{target}' not found in the dataset.")
        return

    # Ensure target is cleanly read as 0 and 1
    df[target] = pd.to_numeric(df[target], errors='coerce').fillna(0).astype(int)

    # Separate features from the target and SMILES names
    non_feat = [target] + (['SMILES'] if 'SMILES' in df.columns else [])
    feat_cols = [c for c in df.columns if c not in non_feat]

    # Automatically detect which features are binary vs continuous
    X_all = df[feat_cols].values.astype(float)
    binary_mask = np.array([
        set(np.unique(X_all[~np.isnan(X_all[:, j]), j])).issubset({0.0, 1.0})
        for j in range(X_all.shape[1])
    ])
    print(f"  Detected {binary_mask.sum()} binary features and {(~binary_mask).sum()} continuous features.")

    # Split the data into positive and negative examples
    pos = df[df[target] == 1].reset_index(drop=True)
    neg = df[df[target] == 0].reset_index(drop=True)
    print(f"\nTarget '{target}':")
    print(f"  Positive : {len(pos)}  ({len(pos)/len(df)*100:.1f}%)")
    print(f"  Negative : {len(neg)}  ({len(neg)/len(df)*100:.1f}%)")

    # Calculate how many new samples we need
    pos_target  = int(args.ratio * len(neg) / (1 - args.ratio))
    n_synthetic = pos_target - len(pos)
    
    if n_synthetic <= 0:
        print("  Already balanced — nothing to do.")
        return
        
    print(f"  Generating {n_synthetic} synthetic positive samples...")

    # Temporarily fill missing values (NaNs) with the median just to do the math
    X = pos[feat_cols].values.astype(float)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        medians = np.nanmedian(X, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        X_imp = X.copy()
        X_imp[nan_mask] = np.take(medians, np.where(nan_mask)[1])
    else:
        X_imp = X

    # Run the SMOTE algorithm
    synth_X = smote_knn(X_imp, n_synthetic, binary_mask, k=args.k)

    # Check the quality of the new data
    stds = np.nanstd(X_imp, axis=0)
    stds[stds == 0] = 1.0
    drift = np.abs((X_imp.mean(0) - synth_X.mean(0)) / stds).mean()
    print(f"  SMOTE quality (normalised drift): {drift:.4f}")

    # Put the synthetic data into a clean table
    synth = pd.DataFrame(synth_X, columns=feat_cols)
    synth[target] = 1
    if 'SMILES' in df.columns:
        synth.insert(0, 'SMILES', 'synthetic_smote')
        
    # Reorder columns to match the original file exactly
    synth = synth[[c for c in df.columns if c in synth.columns]]

    # Combine the old data with the new synthetic data and shuffle it
    balanced = pd.concat([df, synth], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the final file
    out = input_path.parent / (input_path.stem + f"_{target}_balanced.csv")
    balanced.to_csv(out, sep=args.sep, index=False)
    print(f"\nSaved file as: {out.name}")

if __name__ == "__main__":
    main()