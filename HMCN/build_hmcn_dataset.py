"""
build_hmcn_dataset.py
─────────────────────
Builds the input CSV for HMCN-F Option A from two source files:

  - goodscents_jadbio_ready.csv   : 4981 molecules × (1005 features + 12 metacategory labels)
  - Multi-Labelled_Smiles_Odors_dataset.csv : 4983 molecules × 138 fine-grained odor labels

Output: hmcn_dataset.csv
  Columns: SMILES | 1005 features | 138 fine labels (prefix: fine_) | 12 meta labels (prefix: meta_)

Steps
─────
1. Load both source files
2. Drop 5 molecules where ALL Mordred 3D descriptors are NaN (failed 3D embedding)
   — Vassilis's rule: drop rows before dropping columns
3. Inner-join on SMILES  →  4976 molecules
4. Resolve column name conflicts (8 labels exist in both files with same name)
5. Rename fine labels with prefix 'fine_', metacategories with prefix 'meta_'
6. Verify no NaN remains
7. Save to CSV
"""

import pandas as pd
import numpy as np

# ── Paths (adjust if needed) ──────────────────────────────────────────────────
PATH_12  = 'goodscents_jadbio_ready.csv'
PATH_138 = 'Multi-Labelled_Smiles_Odors_dataset.csv'
OUTPUT   = 'hmcn_dataset.csv'

# ── Column definitions ────────────────────────────────────────────────────────
META_LABELS = [
    'floral', 'fruity', 'sweet', 'woody', 'green', 'spicy',
    'animal_musk', 'earthy', 'citrus', 'chemical', 'gourmand', 'powdery_amber'
]

# ── Step 1: Load ──────────────────────────────────────────────────────────────
print("Loading source files...")
df12  = pd.read_csv(PATH_12,  sep=';')
df138 = pd.read_csv(PATH_138)

fine_labels = [c for c in df138.columns if c not in ['nonStereoSMILES', 'descriptors']]
feat_cols   = [c for c in df12.columns  if c not in META_LABELS + ['SMILES']]

print(f"  12-label file  : {df12.shape[0]} molecules, {len(feat_cols)} features")
print(f"  138-label file : {df138.shape[0]} molecules, {len(fine_labels)} fine labels")

# ── Step 2: Drop failed 3D-embedding rows ─────────────────────────────────────
# Mordred 3D descriptors are everything that is not MACCS or Morgan.
# A molecule whose 3D embedding failed has ALL of these as NaN.
# Per Vassilis: drop those rows first, then drop any column still containing NaN.
mordred_cols = [c for c in feat_cols if 'MACCS' not in c and 'morgan' not in c]

failed_3d_mask = df12[mordred_cols].isna().all(axis=1)
n_dropped = failed_3d_mask.sum()
df12 = df12[~failed_3d_mask].copy()
print(f"\nStep 2 — Dropped {n_dropped} molecules with failed 3D embedding")

# Drop any Mordred column still containing a NaN (should be 0 after row drop)
nan_mordred_cols = [c for c in mordred_cols if df12[c].isna().any()]
if nan_mordred_cols:
    df12.drop(columns=nan_mordred_cols, inplace=True)
    feat_cols = [c for c in feat_cols if c not in nan_mordred_cols]
    print(f"         Dropped {len(nan_mordred_cols)} Mordred columns still containing NaN")
else:
    print(f"         No Mordred columns needed dropping after row removal")

print(f"         Remaining: {len(df12)} molecules, {len(feat_cols)} features")

# ── Step 3: Inner join on SMILES ──────────────────────────────────────────────
# 2 molecules in the 138-label file are not in the 12-label file
# (R/S-limonene SMILES bug — known issue). Inner join drops them automatically.
print(f"\nStep 3 — Merging on SMILES (inner join)...")
merged = df12.merge(
    df138,
    left_on='SMILES',
    right_on='nonStereoSMILES',
    how='inner'
)
print(f"         Result: {len(merged)} molecules")

# ── Step 4: Resolve column name conflicts ─────────────────────────────────────
# 8 labels appear in both files with the same name:
# ['citrus','earthy','floral','fruity','green','spicy','sweet','woody']
# Pandas auto-adds _x (from df12) and _y (from df138) suffixes.
# We use _x for metacategories (broader, built by Vassilis's lookup table)
# and _y for fine labels (the original GoodScents/Leffingwell annotations).
overlap = set(META_LABELS) & set(fine_labels)
print(f"\nStep 4 — Resolving {len(overlap)} overlapping column names: {sorted(overlap)}")

# ── Step 5: Build final dataframe ─────────────────────────────────────────────
print("\nStep 5 — Building final dataframe...")

smiles_col = merged[['SMILES']].copy()

# Features (already clean)
features_df = merged[feat_cols].copy()

# Fine labels (138): use _y suffix for overlapping names
fine_df = pd.DataFrame()
for lbl in fine_labels:
    col = lbl + '_y' if lbl in overlap else lbl
    fine_df['fine_' + lbl] = merged[col].values.astype(int)

# Metacategory labels (12): use _x suffix for overlapping names
meta_df = pd.DataFrame()
for lbl in META_LABELS:
    col = lbl + '_x' if lbl in overlap else lbl
    meta_df['meta_' + lbl] = merged[col].values.astype(int)

# Concatenate all at once (avoids DataFrame fragmentation warning)
result = pd.concat([smiles_col.reset_index(drop=True),
                    features_df.reset_index(drop=True),
                    fine_df.reset_index(drop=True),
                    meta_df.reset_index(drop=True)], axis=1)

# ── Step 6: Final verification ────────────────────────────────────────────────
print("\nStep 6 — Verification:")
print(f"  Final shape   : {result.shape}")
print(f"  Expected cols : 1 (SMILES) + {len(feat_cols)} (features)"
      f" + 138 (fine) + 12 (meta) = {1 + len(feat_cols) + 138 + 12}")
print(f"  Any NaN?      : {result.isna().any().any()}")

# Sanity check: for labels that exist in both files,
# the metacategory sum should be >= fine label sum (metacategory is broader)
print("\n  Consistency check (meta_sum >= fine_sum for shared label names):")
for lbl in sorted(overlap):
    meta_sum = result['meta_' + lbl].sum()
    fine_sum = result['fine_' + lbl].sum()
    ok = '✓' if meta_sum >= fine_sum else '✗ WARNING'
    print(f"    {lbl:12s}: meta={meta_sum:4d}, fine={fine_sum:4d}  {ok}")

print("\n  Fine label positives (top 10):")
fine_cols = [c for c in result.columns if c.startswith('fine_')]
print(result[fine_cols].sum().sort_values(ascending=False).head(10).to_string())

print("\n  Meta label positives:")
meta_cols = [c for c in result.columns if c.startswith('meta_')]
print(result[meta_cols].sum().sort_values(ascending=False).to_string())

# ── Step 7: Save ──────────────────────────────────────────────────────────────
result.to_csv(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}")
