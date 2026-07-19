# Changes to `SWEEP_positionalEncoding_GATV2_HMCNF.py`

This document explains every modification made to the original script, why each one was needed, and what to know about it going forward.

---

## 1. Windows local-testing fix (multiprocessing)

**Problem:** running the script directly on Windows crashed with a `multiprocessing`/`spawn` `RuntimeError` inside the `DataLoader`.

**Cause:** Windows starts `DataLoader` worker processes via `spawn`, which re-imports the whole script as a module. Since the script has top-level executable code (not just function/class defs), re-importing it to spawn a worker recursively tried to re-run the whole sweep. Linux (the target server) uses `fork` instead and doesn't have this problem.

**Fix:** an OS-aware constant near the top of the script:

```python
NUM_WORKERS = 0 if os.name == "nt" else 4
PERSISTENT_WORKERS = NUM_WORKERS > 0
```

All `DataLoader(...)` calls now use `num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS` instead of hardcoded values. Windows automatically gets single-process loading (safe, no spawn crash); the Linux server automatically gets 4 parallel workers. No manual toggling needed when moving between machines.

---

## 2. Reproducibility — fixing "same config, different result"

The original code had four independent, uncontrolled sources of randomness that made repeated runs of the *same* configuration give different metrics. Each is now fixed:

### 2a. Global seeding

Near the top of the script:

```python
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
```

Called once at import time, and again at specific points below.

### 2b. The train/val/test split — cached, not just seeded

`create_stratified_splits()` now seeds `np.random` before each `IterativeStratification` call. This matters because `scikit-multilearn`'s `random_state` parameter is **broken in the version installed via `pip`** (no PyPI release since 2018 — the fix exists on GitHub but was never shipped). Its internal tie-breaking silently falls back to the *global* NumPy RNG regardless of what `random_state` you pass, so seeding that global state directly is the only version-independent way to make it reproducible.

We also hit a **second**, unrelated `scikit-multilearn`/`scikit-learn` incompatibility: the installed `scikit-multilearn` hardcodes `shuffle=False` internally, and modern `scikit-learn` raises `ValueError` if `random_state` is set while `shuffle=False`. So `random_state=` was removed from the `IterativeStratification(...)` calls entirely — the `np.random.seed()` workaround above is what actually drives the determinism now.

Beyond seeding, the computed split is **cached to disk**:

```python
SPLIT_CACHE = "fixed_split_indices.pkl"
```

First run computes the split and saves the indices; every run after that loads them from disk instead of recomputing. This sidesteps any remaining RNG/library-version subtlety entirely (including differences between your Windows machine and the Linux server) — the split becomes a fixed artifact, not something regenerated on every run.

### 2c. Model weight initialization

Inside `run_sweep`'s grid loop, `set_seed(SEED)` is called right before `model = SmellGATV2_HMCNF(...)` for **every** configuration in the grid — not just once at the start. Without this, config #5's initial weights would depend on how much random state configs #1–4 had already consumed, making sweep results not properly comparable to each other.

### 2d. DataLoader shuffle order

The training `DataLoader` now gets its own dedicated generator:

```python
generator=torch.Generator().manual_seed(SEED)
```

so batch order is also fixed rather than depending on whatever state the global RNG happened to be in.

### What's *not* covered

GPU-level floating-point non-determinism (from scatter/atomic-add operations in `GATv2Conv` and `global_add_pool`) is not addressed — this is a much smaller residual effect than the four sources above, and fixing it fully (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`) can error out on unsupported ops and slow training. Not implemented; flagged here in case bit-exact reproducibility is ever needed for a paper appendix.

---

## 3. Runtime tracking

Two print statements bookend the script's execution:

```python
SCRIPT_START_TIME = datetime.now()   # right after the reproducibility block
...
elapsed = script_end_time - SCRIPT_START_TIME   # at the very end
```

Prints start time, finish time, and total elapsed duration to the log file — useful for comparing how long different sweep configs cost.

---

## 4. K-fold cross-validation

**Motivation:** with only ~5k molecules, a single fixed train/val/test split can give a metric that's more a reflection of *which molecules happened to land in test* than of the model config itself. The fix above (§2b) makes a single split *reproducible*, but reproducing the same split ten times just reruns the same experiment — it says nothing about how much a config's score would move under a **different** split.

### What was added

- **`build_graphs(df_split)`** — factors the repeated SMILES → PyG-graph loop into a reusable function (previously inlined three times for train/val/test).

- **`run_kfold_sweep(df, k=5, val_ratio_within_train=0.15, ...)`** — the main addition:
  1. **Outer split:** `IterativeStratification(n_splits=k, order=2)` rotates `k` stratified test folds — standard k-fold semantics (full coverage, no overlap between test folds, verified against synthetic data before deployment).
  2. **Inner split:** within each fold's remaining `(k-1)/k` of the data, carves off `val_ratio_within_train` (15% by default) for validation/threshold calibration, using the same `IterativeStratification` machinery as the original single-split code.
  3. **Caching:** all `k` splits are cached to `kfold_results/kfold_split_indices.pkl`, same rationale as §2b — rerunning reuses the same partitions.
  4. **Per fold:** builds graphs, then runs your existing `run_sweep()` + `evaluate_sweep()` pipeline unchanged, with fold-specific output paths (`kfold_results/fold{i}_sweep_results.csv`, `fold{i}_checkpoints/`, `fold{i}_test_results.csv`).
  5. **Aggregation:** combines every fold's test-set row into `kfold_results/all_folds_test_results.csv`, then groups by `(lambda_hier, num_heads, T_0, epochs)` and computes **mean ± std per config** across folds, saved to `kfold_results/kfold_summary_mean_std.csv` — this is the number to actually report.

### Where it's wired in

The bottom of the script now calls `run_kfold_sweep(df=df, k=5, ...)` instead of a single `run_sweep()` + `evaluate_sweep()` call. The original single-split invocation is **commented out, not deleted**, so you can switch back for quick debugging without re-adding it from scratch.

### Important caveat: fold results aren't fully independent

k-fold training sets overlap heavily across folds — for `k=5`, any two folds' training sets share ~60% of their molecules. This means:

- The **mean** across folds is a reasonably honest estimate of expected performance.
- The **std** across folds is real and useful for *comparing configs to each other*, but it **understates the true uncertainty**, because the five fold-level results are correlated rather than independent (Bengio & Grandvalet, 2004). Don't treat it as a rigorous confidence interval.
- Test sets themselves are clean and leak-free *within* each fold — this caveat is about cross-fold correlation, not within-fold leakage.

### `k=5` vs `k=10`

Not changed by default — left at `k=5`. Tradeoffs discussed but not resolved: `k=10` gives more training data per fold and a less noisy std estimate, at exactly 2× the compute for the same grid. Suggested approach if compute is tight: use `k=5` for the hyperparameter search itself, then rerun only the winning config(s) at `k=10` for the number that goes in the writeup.

---

## Summary of files/artifacts this now produces

| File | What it is |
|---|---|
| `fixed_split_indices.pkl` | Cached single-split indices (only used if you switch back to the commented-out single-split path) |
| `kfold_results/kfold_split_indices.pkl` | Cached k-fold partition indices |
| `kfold_results/fold{i}_sweep_results.csv` | Per-fold training/validation results (one row per grid config) |
| `kfold_results/fold{i}_checkpoints/` | Per-fold model checkpoints |
| `kfold_results/fold{i}_test_results.csv` | Per-fold test-set metrics |
| `kfold_results/all_folds_test_results.csv` | All folds' test results combined |
| `kfold_results/kfold_summary_mean_std.csv` | **Mean ± std per config across folds — the headline result** |
