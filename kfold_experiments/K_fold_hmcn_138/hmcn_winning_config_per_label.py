"""
Retrain the single winning HMCN-F config (selected offline as the highest
mean(pr_auc_138) across the 5 folds in hmcn_kfold_ablation_results.csv) on the
5 cached folds, and extract PER-LABEL PR-AUC and F1 for all 138 fine labels --
both the raw per-fold values and a mean +/- std summary across folds.

Why retraining is needed: the k-fold sweep only saved the already-averaged
pr_auc_138 scalar per (config, fold) row -- the per-label PR-AUC values it
computes internally to get that average were never persisted, and no model
checkpoints were saved (600 runs). So the only way to recover a per-label
breakdown for one specific config is to retrain just that config on the 5 folds.

STANDALONE VERSION: this script does NOT import hmcn_kfold_sweep.py. Every
data-loading / fold-splitting / model / training-loop building block it needs
(load_raw, build_pairs, build_or_load_kfold_splits, build_fold_data, Block,
HMCNF, bce, vloss, loss_fn, collect, mauc, set_seed) is copied in below,
verbatim from hmcn_kfold_sweep.py, so the retrained folds are still identical
to what the original sweep ran. It still depends on:
  - hmcn_eval.py       (for find_optimal_thresholds, same convention as the sweep)
  - hmcn_dataset.csv   (the data)
  - kfold_results/hmcn_kfold_split_indices.pkl  (optional -- reused if present,
    otherwise recomputed fresh and cached under that same path)

PR-AUC is threshold-independent, so per-label PR-AUC does not depend on the
val-calibrated fine_thresholds. F1 is threshold-dependent, so it's computed at
each label's own val-calibrated threshold, same convention as hmcn_eval.
"""

import os
import csv
import random
import pickle
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    precision_score, recall_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix,
)
from skmultilearn.model_selection import IterativeStratification
import hmcn_eval  # shared eval utility -- place hmcn_eval.py in the same directory

# ============================== CONFIG ==========================================
CSV_PATH     = 'hmcn_dataset.csv'
RESULTS_DIR  = 'kfold_results'
SPLIT_CACHE  = os.path.join(RESULTS_DIR, 'hmcn_kfold_split_indices.pkl')
PER_FOLD_CSV = os.path.join(RESULTS_DIR, 'hmcn_winning_config_fine138_per_label_per_fold.csv')
SUMMARY_CSV  = os.path.join(RESULTS_DIR, 'hmcn_winning_config_fine138_per_label_summary.csv')
LOG_PATH     = os.path.join(RESULTS_DIR, 'hmcn_winning_config_per_label.log')

K            = 5
VAL_RATIO    = 0.15
EPOCHS       = 120
PATIENCE     = 20
BATCH_SIZE   = 32
SEED         = 42
DEBUG        = False   # True -> tiny data slice, 2 epochs, for a smoke test

# Winning config, selected offline as the highest mean(pr_auc_138) across the 5
# folds in hmcn_kfold_ablation_results.csv.
WINNING_CFG = {
    'gd': 128,    # global_dim
    'ld': 64,     # local_dim
    'dr': 0.5,    # dropout
    'lr': 1e-3,
    'wd': 1e-4,   # weight_decay
    'lv': 0.0,    # lambda_viol
}
BETA = 0.5

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
# ==================================================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

logger = logging.getLogger('hmcn_winning_config')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_PATH)
file_handler.setFormatter(logging.Formatter('%(asctime)s  %(message)s'))
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())  # also echo to stdout under nohup -u


# --- Reproducibility: fix all sources of randomness ------------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)          # also fixes skmultilearn's IterativeStratification,
                                   # whose internal tie-breaking falls back to the
                                   # global numpy RNG regardless of random_state
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
# -----------------------------------------------------------------------------------


# ============================== DATA (copied from hmcn_kfold_sweep.py) ============

def load_raw(csv_path=CSV_PATH):
    """Load full feature/label matrices. No filtering, no scaling -- that
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
        logger.info(f"Loaded cached {len(fold_splits)}-fold split from '{cache_path}'.")
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
    logger.info(f"Computed a fresh {k}-fold split and cached it to '{cache_path}'.")
    return fold_splits


def build_fold_data(X, Y1, Y2, train_idx, val_idx, test_idx):
    """Zero-variance filtering AND StandardScaler fitting done here, on the
    fold's train split only -- not globally -- to avoid val/test leakage."""
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


# ============================== MODEL (copied from hmcn_kfold_sweep.py) ===========

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


def mauc(p, t):
    """Per-epoch early-stopping signal only -- macro ROC-AUC, no thresholding involved."""
    return np.mean([roc_auc_score(t[:, i], p[:, i]) for i in range(t.shape[1]) if t[:, i].sum() > 0])


# ============================== TRAINING (adapted from hmcn_kfold_sweep.trial) ====

def train_fold_and_collect(cfg, data, pairs, device, epochs=EPOCHS, patience=PATIENCE, seed=SEED):
    """Same training loop as hmcn_kfold_sweep.trial(), but additionally returns
    the raw test-set fine predictions/targets (fp_t, ft_t) needed for per-label
    PR-AUC, plus the val-calibrated fine_thresholds needed for per-label F1."""
    Xtr, Y1tr, Y2tr, Xva, Y1va, Y2va, Xte, Y1te, Y2te = data

    def T(*a):
        return [torch.tensor(x, dtype=torch.float32) for x in a]

    set_seed(seed)
    trl = DataLoader(TensorDataset(*T(Xtr, Y1tr, Y2tr)), batch_size=BATCH_SIZE, shuffle=True,
                      generator=torch.Generator().manual_seed(seed))
    val = DataLoader(TensorDataset(*T(Xva, Y1va, Y2va)), batch_size=128)
    tel = DataLoader(TensorDataset(*T(Xte, Y1te, Y2te)), batch_size=128)

    set_seed(seed)  # reseed again immediately before model init -- controls weight init
    m = HMCNF(Xtr.shape[1], Y1tr.shape[1], Y2tr.shape[1],
              cfg['gd'], cfg['ld'], cfg['dr'], BETA).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    best_auc, best_st, best_epoch, cnt = 0., None, 0, 0
    for ep in range(1, epochs + 1):
        m.train()
        for Xb, Y1b, Y2b in trl:
            Xb, Y1b, Y2b = Xb.to(device), Y1b.to(device), Y2b.to(device)
            opt.zero_grad()
            PF, P1, P2, PG = m(Xb)
            loss = loss_fn(PF, P1, P2, PG, Y1b, Y2b, pairs, cfg['lv'])
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.)
            opt.step()

        fp_v, ft_v, mp_v, mt_v = collect(m, val, device)
        auc = mauc(mp_v, mt_v)
        sched.step(-auc)
        if auc > best_auc:
            best_auc = auc
            best_epoch = ep
            best_st = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            cnt = 0
        else:
            cnt += 1
            if cnt >= patience:
                break

        if DEBUG and ep >= 2:
            break

    m.load_state_dict(best_st)
    m.to(device)

    # Threshold calibration reuses hmcn_eval, same convention as the sweep.
    fp_v, ft_v, mp_v, mt_v = collect(m, val, device)
    fine_thresholds = hmcn_eval.find_optimal_thresholds(fp_v, ft_v)

    fp_t, ft_t, mp_t, mt_t = collect(m, tel, device)

    return fp_t, ft_t, fine_thresholds, best_epoch, best_auc


def compute_label_metrics(y_true, y_prob, threshold):
    """Full per-label metric set at a given (val-calibrated) threshold.
    Returns a dict; entries that are structurally undefined for this fold/label
    (e.g. ROC_AUC with only one class present) are set to NaN rather than
    silently defaulting to 0, so they're visibly distinguishable in the CSV."""
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    roc_auc = roc_auc_score(y_true, y_prob) if (n_pos > 0 and n_neg > 0) else float('nan')
    pr_auc = average_precision_score(y_true, y_prob) if n_pos > 0 else float('nan')

    mcc_defined = (tp + fp) and (tp + fn) and (tn + fp) and (tn + fn)

    return {
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'Bal_Acc': balanced_accuracy_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred) if mcc_defined else float('nan'),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Sensitivity': recall_score(y_true, y_pred, zero_division=0),  # = TPR = TP/(TP+FN)
        'Specificity': (tn / (tn + fp)) if (tn + fp) > 0 else float('nan'),
    }


# ============================== MAIN ================================================

def run():
    script_start = datetime.now()
    logger.info(f"Script started at: {script_start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Winning config: {WINNING_CFG}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    X, Y1, Y2, fine_names, meta_names = load_raw(CSV_PATH)
    pairs = build_pairs(fine_names, meta_names)
    fold_splits = build_or_load_kfold_splits(X, Y1, k=K, val_ratio=VAL_RATIO,
                                              seed=SEED, cache_path=SPLIT_CACHE)
    n_labels = len(fine_names)

    # One row per (label, fold) -- the raw, un-aggregated results.
    per_fold_rows = []

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(fold_splits):
        logger.info(f"--- Fold {fold_i + 1}/{len(fold_splits)} "
                    f"({len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test) ---")

        data = build_fold_data(X, Y1, Y2, train_idx, val_idx, test_idx)
        fp_t, ft_t, fine_thresholds, best_epoch, best_auc = train_fold_and_collect(
            WINNING_CFG, data, pairs, device, epochs=EPOCHS, patience=PATIENCE, seed=SEED,
        )
        logger.info(f"  best_epoch={best_epoch}  val_meta_roc_auc={best_auc:.4f}")

        for i in range(n_labels):
            y_true = ft_t[:, i]
            y_prob = fp_t[:, i]
            support = int(y_true.sum())

            if support == 0:
                logger.warning(f"  Label '{fine_names[i]}' has 0 positives in the "
                                f"fold {fold_i} test split -- recording NaN metrics for this fold.")
                per_fold_rows.append({
                    'label': fine_names[i], 'fold': fold_i, 'support': support,
                    'TP': float('nan'), 'TN': float('nan'), 'FP': float('nan'), 'FN': float('nan'),
                    'Bal_Acc': float('nan'), 'MCC': float('nan'), 'F1': float('nan'),
                    'ROC_AUC': float('nan'), 'PR_AUC': float('nan'),
                    'Precision': float('nan'), 'Sensitivity': float('nan'), 'Specificity': float('nan'),
                })
                continue

            m = compute_label_metrics(y_true, y_prob, fine_thresholds[i])
            row = {'label': fine_names[i], 'fold': fold_i, 'support': support}
            row.update(m)
            per_fold_rows.append(row)

    # --- Raw per-fold table: 138 labels x 5 folds = up to 690 rows ---
    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv(PER_FOLD_CSV, index=False)
    logger.info(f"Per-fold, per-label table saved to '{PER_FOLD_CSV}' "
                f"({len(per_fold_df)} rows = {n_labels} labels x {len(fold_splits)} folds)")

    # --- Summary table: mean/std across folds, computed FROM the per-fold table ---
    def population_std(s):
        # ddof=0, matching np.std's default used elsewhere in the project
        # (pandas' .std() defaults to ddof=1, the sample std -- deliberately overridden here)
        return s.std(ddof=0)

    metric_cols = ['TP', 'TN', 'FP', 'FN', 'Bal_Acc', 'MCC', 'F1',
                    'ROC_AUC', 'PR_AUC', 'Precision', 'Sensitivity', 'Specificity']

    agg_spec = {'n_folds_evaluated': ('PR_AUC', 'count'),   # NaN folds excluded by pandas
                'support_mean': ('support', 'mean')}
    for col in metric_cols:
        agg_spec[f'{col}_mean'] = (col, 'mean')
        agg_spec[f'{col}_std'] = (col, population_std)

    summary_df = (
        per_fold_df.groupby('label')
        .agg(**agg_spec)
        .reset_index()
        .sort_values('PR_AUC_mean', ascending=False)
    )
    round_cols = {c: 4 for c in summary_df.columns if c.endswith('_mean') or c.endswith('_std')}
    round_cols['support_mean'] = 1
    summary_df = summary_df.round(round_cols)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    logger.info(f"Summary (mean +/- std across folds) table saved to '{SUMMARY_CSV}'")

    macro_pr = summary_df['PR_AUC_mean'].mean()
    macro_f1 = summary_df['F1_mean'].mean()
    logger.info(f"Sanity check -- macro mean of per-label PR_AUC_mean: {macro_pr:.4f} "
                f"(should be close to the winning config's mean pr_auc_138 from the sweep)")
    logger.info(f"Macro mean F1 (per-label, val-calibrated thresholds): {macro_f1:.4f}")

    script_end = datetime.now()
    logger.info(f"Script finished at: {script_end.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total elapsed time: {script_end - script_start}")

    return per_fold_df, summary_df


if __name__ == '__main__':
    run()
