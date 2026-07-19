"""
HMCN-F k-fold cross-validated hyperparameter sweep.

Adapted from Vinicius's SWEEP_k_fold_positionalEncoding_GATV2.py reproducibility
pattern, applied to the tabular-feature (non-graph) HMCN-F track.

Key differences from his script:
  - No graph construction — features are already flat rows in hmcn_dataset.csv.
  - Stratification is on Y1 (138 fine labels), matching this track's established
    convention, not the 12 macro/meta labels.
  - Zero-variance feature dropping and StandardScaler fitting happen INSIDE each
    fold, on that fold's train split only — not once globally — to avoid leaking
    val/test statistics into the train normalization (agreed decision, see chat).
"""

import os
import csv
import random
import pickle
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score , average_precision_score
from skmultilearn.model_selection import IterativeStratification
import hmcn_eval  # shared eval utility — place hmcn_eval.py in the same directory

# hmcn_eval.save_experiment() only writes columns in _CSV_COLUMNS; extend it with
# 'fold' so per-fold rows are distinguishable (needed for the mean/std aggregation
# below). This is the one deliberate, intentional mutation of that module's schema.
if 'fold' not in hmcn_eval._CSV_COLUMNS:
    hmcn_eval._CSV_COLUMNS.insert(3, 'fold')

# --- Reproducibility: fix all sources of randomness ---------------------------
SEED = 42

def set_seed(seed=SEED):
    """Seed every RNG that affects this pipeline's stochastic behaviour."""
    random.seed(seed)
    np.random.seed(seed)          # also fixes skmultilearn's IterativeStratification,
                                   # whose internal tie-breaking falls back to the
                                   # global numpy RNG regardless of random_state
                                   # (https://github.com/scikit-multilearn/scikit-multilearn/issues/144)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
# -------------------------------------------------------------------------------

# --- Runtime tracking -----------------------------------------------------------
SCRIPT_START_TIME = datetime.now()
print(f"Script started at: {SCRIPT_START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
# ---------------------------------------------------------------------------------

# ============================== CONFIG ==========================================
CSV_PATH        = 'hmcn_dataset.csv'
K               = 5
VAL_RATIO       = 0.15          # within each fold's train+val portion
RESULTS_DIR     = 'kfold_results'
SPLIT_CACHE     = os.path.join(RESULTS_DIR, 'hmcn_kfold_split_indices.pkl')
EPOCHS          = 120
PATIENCE        = 20
BATCH_SIZE      = 32
DEBUG           = False         # True -> tiny data slice, 2 epochs, for a smoke test

GRID = {
    'gd': [64, 128],
    'ld': [32, 64],
    'dr': [0.5, 0.6, 0.7],
    'lr': [1e-3],
    'wd': [1e-4, 1e-3],
    'lv': [0, 0.01, 0.05, 0.1, 0.3],
}
# ==================================================================================

META_CATEGORIES = {
    'floral': ['floral','rose','jasmin','lily','muguet','violet','hyacinth','geranium','lavender','orangeflower','chamomile','hawthorn'],
    'fruity': ['fruity','apple','apricot','banana','berry','cherry','grape','grapefruit','lemon','melon','orange','peach','pear','pineapple','plum','raspberry','strawberry','tropical','black currant','fruit skin'],
    'sweet': ['sweet','vanilla','caramellic','honey','chocolate','cocoa','coconut','creamy','buttery','milky','dairy'],
    'woody': ['woody','cedar','sandalwood','pine','vetiver','terpenic','balsamic','cortex'],
    'green': ['green','grassy','herbal','leafy','hay','tea','fresh','cucumber','vegetable','weedy'],
    'spicy': ['spicy','cinnamon','clove','warm','pungent','sharp','cooling','mint','camphoreous'],
    'animal_musk': ['animal','musk','leathery','fishy','sweaty','meaty','beefy','musty'],
    'earthy': ['earthy','mushroom','nutty','hazelnut','roasted','coffee','tobacco','smoky','popcorn'],
    'citrus': ['citrus','bergamot','ozone','clean','soapy'],
    'chemical': ['solvent','ethereal','metallic','medicinal','phenolic','sulfurous','gassy','burnt','oily'],
    'gourmand': ['almond','malty','rummy','brandy','cognac','winey','cooked','potato','savory','celery','tomato','radish','onion','garlic','cabbage','cheesy'],
    'powdery_amber': ['amber','powdery','anisic','coumarinic','orris','waxy','aldehydic','ketonic','lactonic'],
}


# ============================== DATA ==============================================

def load_raw(csv_path=CSV_PATH):
    """Load full feature/label matrices. No filtering, no scaling — that
    happens per fold, on the train split only, in build_fold_data()."""
    df = pd.read_csv(csv_path)
    fine_cols = [c for c in df.columns if c.startswith('fine_')]
    meta_cols = [c for c in df.columns if c.startswith('meta_')]
    feat_cols = [c for c in df.columns if c not in fine_cols + meta_cols + ['SMILES']]

    X  = df[feat_cols].values.astype(np.float32)
    Y1 = df[fine_cols].values.astype(np.float32)   # 138 fine labels
    Y2 = df[meta_cols].values.astype(np.float32)   # 12 metacategories

    fine_names = [c.replace('fine_', '') for c in fine_cols]
    meta_names = [c.replace('meta_', '') for c in meta_cols]

    if DEBUG:
        rng = np.random.RandomState(SEED)
        sub = rng.choice(len(X), size=min(300, len(X)), replace=False)
        X, Y1, Y2 = X[sub], Y1[sub], Y2[sub]

    return X, Y1, Y2, fine_names, meta_names


def build_pairs(fine_names, meta_names):
    fine_idx = {n: i for i, n in enumerate(fine_names)}
    meta_idx = {n: i for i, n in enumerate(meta_names)}
    return [(fine_idx[m], meta_idx[meta])
            for meta, members in META_CATEGORIES.items()
            for m in members if m in fine_idx and meta in meta_idx]


def build_or_load_kfold_splits(X, Y1, k=K, val_ratio=VAL_RATIO, seed=SEED,
                                cache_path=SPLIT_CACHE):
    """Returns a list of k (train_idx, val_idx, test_idx) tuples, cached to disk.
    Rerunning this function reuses the exact same partitions."""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            fold_splits = pickle.load(f)
        print(f"Loaded cached {len(fold_splits)}-fold split from '{cache_path}'.")
        return fold_splits

    placeholder = np.arange(len(X)).reshape(-1, 1)
    np.random.seed(seed)
    outer = IterativeStratification(n_splits=k, order=2)
    outer_folds = list(outer.split(placeholder, Y1))  # stratify on Y1, per project convention

    fold_splits = []
    for fold_i, (trainval_idx, test_idx) in enumerate(outer_folds):
        np.random.seed(seed + fold_i + 1)   # distinct-but-deterministic per fold
        X_tv = np.arange(len(trainval_idx)).reshape(-1, 1)
        y_tv = Y1[trainval_idx]
        inner = IterativeStratification(
            n_splits=2, order=2,
            sample_distribution_per_fold=[val_ratio, 1 - val_ratio],
        )
        train_rel, val_rel = next(inner.split(X_tv, y_tv))
        train_idx = trainval_idx[train_rel]
        val_idx   = trainval_idx[val_rel]
        fold_splits.append((train_idx, val_idx, test_idx))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(fold_splits, f)
    print(f"Computed a fresh {k}-fold split and cached it to '{cache_path}'.")
    return fold_splits


def build_fold_data(X, Y1, Y2, train_idx, val_idx, test_idx):
    """Zero-variance filtering AND StandardScaler fitting done here, on the
    fold's train split only — not globally — to avoid val/test leakage."""
    Xtr_raw, Xva_raw, Xte_raw = X[train_idx], X[val_idx], X[test_idx]

    stds = Xtr_raw.std(axis=0)
    keep = stds > 0
    Xtr_raw, Xva_raw, Xte_raw = Xtr_raw[:, keep], Xva_raw[:, keep], Xte_raw[:, keep]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_raw)
    Xva = scaler.transform(Xva_raw)
    Xte = scaler.transform(Xte_raw)

    return (Xtr, Y1[train_idx], Y2[train_idx],
            Xva, Y1[val_idx],   Y2[val_idx],
            Xte, Y1[test_idx],  Y2[test_idx])


# ============================== MODEL ==============================================

class Block(nn.Module):
    def __init__(self, idim, gdim, ldim, nl, dr):
        super().__init__()
        self.gfc = nn.Sequential(nn.Linear(gdim + idim, gdim), nn.BatchNorm1d(gdim), nn.ReLU(), nn.Dropout(dr))
        self.tr  = nn.Sequential(nn.Linear(gdim, ldim), nn.ReLU(), nn.Dropout(dr))
        self.out = nn.Linear(ldim, nl)

    def forward(self, x, A):
        A = self.gfc(torch.cat([A, x], 1))
        return A, torch.sigmoid(self.out(self.tr(A)))


class HMCNF(nn.Module):
    def __init__(self, idim, nf, nm, gd, ld, dr, beta):
        super().__init__()
        self.beta = beta
        self.proj = nn.Sequential(nn.Linear(idim, gd), nn.BatchNorm1d(gd), nn.ReLU(), nn.Dropout(dr))
        self.l1 = Block(idim, gd, ld, nf, dr)
        self.l2 = Block(idim, gd, ld, nm, dr)
        self.go = nn.Linear(gd, nf + nm)

    def forward(self, x):
        A = self.proj(x)
        A, P1 = self.l1(x, A)
        A, P2 = self.l2(x, A)
        PG = torch.sigmoid(self.go(A))
        PF = self.beta * torch.cat([P1, P2], 1) + (1 - self.beta) * PG
        return PF, P1, P2, PG


def bce(P, Y, e=1e-7):
    P = torch.clamp(P, e, 1 - e)
    return -torch.mean(Y * torch.log(P) + (1 - Y) * torch.log(1 - P))

def vloss(P1, P2, pairs):
    L = torch.tensor(0., device=P1.device)
    for fi, mi in pairs:
        L = L + torch.mean(torch.clamp(P1[:, fi] - P2[:, mi], min=0) ** 2)
    return L / max(len(pairs), 1)

def loss_fn(PF, P1, P2, PG, Y1, Y2, pairs, lv):
    return bce(P1, Y1) + bce(P2, Y2) + bce(PG, torch.cat([Y1, Y2], 1)) + lv * vloss(P1, P2, pairs)


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    fp, ft, mp, mt = [], [], [], []
    for Xb, Y1b, Y2b in loader:
        _, P1, P2, _ = model(Xb.to(device))
        fp.append(P1.cpu().numpy()); ft.append(Y1b.numpy())
        mp.append(P2.cpu().numpy()); mt.append(Y2b.numpy())
    return np.vstack(fp), np.vstack(ft), np.vstack(mp), np.vstack(mt)


def macro_pr_auc(p, t):
    """Macro-averaged PR-AUC (average precision) across labels with at least
    one positive in this split. Used as the early-stopping / checkpoint signal."""
    return np.mean([average_precision_score(t[:, i], p[:, i])
                     for i in range(t.shape[1]) if t[:, i].sum() > 0])



# ============================== TRIAL ==============================================

def trial(cfg, data, pairs, fine_names, meta_names, device, epochs=EPOCHS, patience=PATIENCE, seed=SEED):
    Xtr, Y1tr, Y2tr, Xva, Y1va, Y2va, Xte, Y1te, Y2te = data

    def T(*a):
        return [torch.tensor(x, dtype=torch.float32) for x in a]

    # Reset RNG so every config starts from the same initial weights, and the
    # train loader's shuffle order is fixed rather than depending on whatever
    # global RNG state the previous config left behind.
    set_seed(seed)
    trl = DataLoader(TensorDataset(*T(Xtr, Y1tr, Y2tr)), batch_size=BATCH_SIZE, shuffle=True,
                      generator=torch.Generator().manual_seed(seed))
    val = DataLoader(TensorDataset(*T(Xva, Y1va, Y2va)), batch_size=128)
    tel = DataLoader(TensorDataset(*T(Xte, Y1te, Y2te)), batch_size=128)

    set_seed(seed)  # reseed again immediately before model init — this is what actually controls weight init
    m = HMCNF(Xtr.shape[1], Y1tr.shape[1], Y2tr.shape[1],
              cfg['gd'], cfg['ld'], cfg['dr'], 0.5).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    best_auc, best_st, best_epoch, best_train_loss, cnt = 0., None, 0, float('nan'), 0
    for ep in range(1, epochs + 1):
        m.train()
        epoch_loss, n_batches = 0., 0
        for Xb, Y1b, Y2b in trl:
            Xb, Y1b, Y2b = Xb.to(device), Y1b.to(device), Y2b.to(device)
            opt.zero_grad()
            PF, P1, P2, PG = m(Xb)
            loss = loss_fn(PF, P1, P2, PG, Y1b, Y2b, pairs, cfg['lv'])
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.)
            opt.step()
            epoch_loss += loss.item(); n_batches += 1
        epoch_loss /= max(n_batches, 1)

        fp_v, ft_v, mp_v, mt_v = collect(m, val, device)
        fine_pr_auc = macro_pr_auc(fp_v, ft_v)
        meta_pr_auc = macro_pr_auc(mp_v, mt_v)
        auc = (fine_pr_auc + meta_pr_auc) / 2   # combined signal drives stopping/checkpointing

        sched.step(-auc)
        if auc > best_auc:
            best_auc = auc
            best_epoch = ep
            best_train_loss = epoch_loss
            best_fine_pr_auc = fine_pr_auc
            best_meta_pr_auc = meta_pr_auc
            best_st = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            cnt = 0
        else:
            cnt += 1
            if cnt >= patience:
                break

        if DEBUG and ep >= 2:
            break

    m.load_state_dict(best_st); m.to(device)

    # --- Final metrics: thresholds calibrated on val, everything else via hmcn_eval ---
    fp_v, ft_v, mp_v, mt_v = collect(m, val, device)
    fine_thresholds = hmcn_eval.find_optimal_thresholds(fp_v, ft_v)
    meta_thresholds = hmcn_eval.find_optimal_thresholds(mp_v, mt_v)

    fp_t, ft_t, mp_t, mt_t = collect(m, tel, device)
    test_loss = hmcn_eval.compute_test_loss(m, tel, pairs, cfg['lv'], device)

    metrics = hmcn_eval.compute_all_metrics(
        fine_probs=fp_t, fine_true=ft_t,
        meta_probs=mp_t, meta_true=mt_t,
        meta_thresholds=meta_thresholds, fine_thresholds=fine_thresholds,
        violation_pairs=pairs, meta_names=meta_names, fine_names=fine_names,
        Y2_train=Y2tr, test_loss=test_loss,
    )

    run_info = {
        'best_epoch': best_epoch,
        'train_loss_at_best': round(float(best_train_loss), 4),
        'val_fine_pr_auc': round(float(best_fine_pr_auc), 4),
        'val_meta_pr_auc': round(float(best_meta_pr_auc), 4),
        'val_combined_pr_auc': round(float(best_auc), 4),
    }
    return run_info, metrics


# ============================== K-FOLD SWEEP ========================================

def run_kfold_sweep(csv_path=CSV_PATH, k=K, val_ratio=VAL_RATIO, grid=GRID,
                     epochs=EPOCHS, patience=PATIENCE, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X, Y1, Y2, fine_names, meta_names = load_raw(csv_path)
    pairs = build_pairs(fine_names, meta_names)
    fold_splits = build_or_load_kfold_splits(X, Y1, k=k, val_ratio=val_ratio)

    keys = list(grid.keys())  # gd, ld, dr, lr, wd, lv
    configs = [dict(zip(keys, c)) for c in itertools.product(*grid.values())]
    print(f"Configs per fold: {len(configs)} | Folds: {len(fold_splits)} | "
          f"Total runs: {len(configs) * len(fold_splits)}\n")

    all_rows_path = os.path.join(results_dir, 'hmcn_kfold_ablation_results.csv')
    # maps grid keys -> the column names hmcn_eval's schema actually uses
    COL = {'gd': 'global_dim', 'ld': 'local_dim', 'dr': 'dropout',
           'lr': 'lr', 'wd': 'weight_decay', 'lv': 'lambda_viol'}

    # --- Restart-skip: figure out which (fold, config) pairs are already done ---
    def cfg_key(fold_i, cfg):
        return (fold_i,) + tuple(round(float(cfg[k]), 8) for k in keys)

    completed = set()
    if os.path.exists(all_rows_path):
        prev = pd.read_csv(all_rows_path)
        for _, row in prev.iterrows():
            completed.add((int(row['fold']),) + tuple(round(float(row[COL[k]]), 8) for k in keys))
        print(f"Found {len(completed)} completed runs in '{all_rows_path}' — these will be skipped.")
    # -----------------------------------------------------------------------------

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(fold_splits):
        print(f"\n{'='*60}\nFOLD {fold_i+1}/{len(fold_splits)}  "
              f"({len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test)\n{'='*60}")

        fold_configs_remaining = [cfg for cfg in configs if cfg_key(fold_i, cfg) not in completed]
        if not fold_configs_remaining:
            print(f"  All {len(configs)} configs already completed for this fold — skipping.")
            continue

        data = build_fold_data(X, Y1, Y2, train_idx, val_idx, test_idx)
        Y2tr = data[2]  # this fold's train meta labels, needed for cooc-consistency reference

        for i, cfg in enumerate(configs, 1):
            if cfg_key(fold_i, cfg) in completed:
                print(f"  [{i}/{len(configs)}] {cfg} -> already completed, skipping")
                continue

            run_info, metrics = trial(cfg, data, pairs, fine_names, meta_names, device,
                                       epochs=epochs, patience=patience)

            param_value = (f"gd={cfg['gd']}_ld={cfg['ld']}_do={cfg['dr']}_"
                            f"lr={cfg['lr']:.0e}_wd={cfg['wd']:.0e}_lv={cfg['lv']}")
            config = dict(
                experiment='hmcn_kfold_sweep', param_name='grid', param_value=param_value,
                fold=fold_i,
                global_dim=cfg['gd'], local_dim=cfg['ld'], dropout=cfg['dr'],
                lr=cfg['lr'], weight_decay=cfg['wd'], lambda_viol=cfg['lv'], beta=0.5,
                batch_size=BATCH_SIZE, seed=SEED,
                **run_info,
            )
            hmcn_eval.save_experiment(config, metrics, csv_path=all_rows_path)
            print(f"  [{i}/{len(configs)}] {cfg} -> "
                  f"fine_pr_auc={metrics['pr_auc_138']:.4f} fine_f1={metrics['f1_macro_138']:.4f} "
                  f"meta_pr_auc={metrics['pr_auc_12']:.4f} meta_f1={metrics['f1_macro_12']:.4f}")

    combined = pd.read_csv(all_rows_path)
    metric_cols = [c for c in hmcn_eval._CSV_COLUMNS
                   if c not in ('experiment', 'param_name', 'param_value', 'fold',
                                'global_dim', 'local_dim', 'dropout', 'lr', 'weight_decay',
                                'lambda_viol', 'beta', 'batch_size', 'seed')]
    summary = combined.groupby(['global_dim', 'local_dim', 'dropout', 'lr', 'weight_decay',
                                 'lambda_viol'])[metric_cols].agg(['mean', 'std'])
    summary_path = os.path.join(results_dir, 'hmcn_kfold_summary_mean_std.csv')
    summary.to_csv(summary_path)

    print(f"\n{'='*60}\nK-fold sweep complete.")
    print(f"  Per-run results (same schema as hmcn_ablation_results.csv, plus 'fold'): '{all_rows_path}'")
    print(f"  Mean ± std summary (report this): '{summary_path}'")
    print(f"{'='*60}")
    return combined, summary


if __name__ == '__main__':
    run_kfold_sweep()

    script_end_time = datetime.now()
    elapsed = script_end_time - SCRIPT_START_TIME
    print(f"\nScript finished at: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed time: {elapsed}")
