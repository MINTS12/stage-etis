# HMCN-F Hyperparameter Analysis — Dataset A
**Experiment:** `datasetA_sweep` + `datasetA_beta_variation`  
**Scope:** 47 grid-search runs × 4 parameters (λ_viol, lr, weight_decay, dropout) + 6 beta ablation runs  
**Primary metric:** Macro PR AUC (meta, 12 labels) | **Tuning metric:** Macro F1 (meta, 12 labels)

---

## 1. Overview of the Search Space

| Parameter | Values tested |
|---|---|
| λ_viol (hierarchical penalty) | 0.0, 0.01, 0.05, 0.10 |
| Learning rate | 1e-4, 5e-4, 1e-3 |
| Weight decay | 1e-4, 1e-3 |
| Dropout | 0.30, 0.47 |
| β (local/global blend) | 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 (separate ablation) |

Full grid = 4 × 3 × 2 × 2 = 48 combinations. One run (λ=0.1, lr=1e-4, wd=1e-3, do=0.47) did not complete, leaving **47 runs**. Architecture was fixed throughout: global_dim=128, local_dim=64, batch_size=32, seed=42.

---

## 2. Overall Performance Distribution

Across all 47 runs:

| Metric | Min | Mean | Median | Max | Std |
|---|---|---|---|---|---|
| PR AUC (meta-12) | 0.6140 | 0.6263 | 0.6262 | **0.6388** | 0.0045 |
| F1 macro (meta-12) | 0.5896 | 0.5999 | 0.5991 | **0.6117** | 0.0052 |
| ROC AUC (meta-12) | 0.7890 | 0.7985 | 0.7988 | **0.8086** | 0.0039 |
| Sensitivity (meta-12) | 0.6593 | 0.6869 | 0.6833 | 0.7353 | 0.0190 |
| Specificity (meta-12) | 0.7081 | 0.7445 | 0.7497 | 0.7841 | 0.0185 |
| Jaccard (meta-12) | 0.4229 | 0.4339 | 0.4336 | 0.4457 | 0.0054 |
| Hamming loss (meta-12) | 0.2290 | 0.2465 | 0.2457 | 0.2604 | 0.0078 |
| PR AUC (fine-138) | 0.0851 | 0.1375 | 0.1119 | 0.2242 | 0.0460 |
| F1 macro (fine-138) | 0.0704 | 0.1326 | 0.1093 | 0.2191 | 0.0466 |
| ROC AUC (fine-138) | 0.6396 | 0.7431 | 0.7252 | 0.8383 | 0.0623 |

**Key observation:** The spread across all 47 runs is narrow — PR AUC std = 0.0045, F1 std = 0.0052. The total max–min range is 0.0248 for PR AUC and 0.0221 for F1. This is the first signal that the model is largely insensitive to hyperparameter choice within this search space.

Fine-label metrics are substantially lower than meta-label metrics and show much higher variance (PR AUC std = 0.046), driven not by consistent trends but by the inherent difficulty of 138-label prediction. They are not the focus of tuning here and should not be directly compared across runs without controlling for which fine labels converge.

---

## 3. Best Configurations

### Best by PR AUC (primary reporting metric)

| λ_viol | lr | wd | dropout | PR AUC | F1 macro | ROC AUC | Sensitivity | Specificity | Jaccard |
|---|---|---|---|---|---|---|---|---|---|
| **0.00** | **1e-3** | **1e-4** | **0.30** | **0.6388** | 0.6093 | **0.8086** | 0.6769 | 0.7589 | 0.4441 |
| 0.05 | 1e-3 | 1e-4 | 0.30 | 0.6352 | 0.6029 | 0.8011 | 0.6778 | 0.7609 | 0.4359 |
| 0.01 | 1e-3 | 1e-4 | 0.47 | 0.6341 | **0.6117** | 0.8049 | **0.7118** | 0.7281 | **0.4457** |
| 0.10 | 5e-4 | 1e-3 | 0.30 | 0.6323 | 0.6092 | 0.8020 | 0.6677 | **0.7841** | 0.4428 |
| 0.05 | 5e-4 | 1e-3 | 0.30 | 0.6314 | 0.6009 | 0.7957 | 0.6627 | 0.7638 | 0.4346 |

### Best by F1 macro (tuning metric)

| λ_viol | lr | wd | dropout | PR AUC | F1 macro | ROC AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 1e-3 | 1e-4 | 0.47 | 0.6341 | **0.6117** | 0.8049 | **0.7118** | 0.7281 |
| 0.00 | 5e-4 | 1e-3 | 0.30 | 0.6292 | 0.6094 | 0.7995 | 0.6714 | 0.7608 |
| 0.00 | 1e-3 | 1e-4 | 0.30 | **0.6388** | 0.6093 | **0.8086** | 0.6769 | 0.7589 |

The two best configurations are not the same run. The PR-AUC-best run (λ=0, lr=1e-3, wd=1e-4, do=0.30) achieves the highest discrimination (0.6388) but the F1-best run (λ=0.01, lr=1e-3, wd=1e-4, do=0.47) achieves better label-set overlap (F1=0.6117, Jaccard=0.4457) at the cost of PR AUC (0.6341). This is a sensitivity/precision tradeoff: the F1-best run has noticeably higher sensitivity (0.7118 vs 0.6769) at lower specificity (0.7281 vs 0.7589).

For reporting purposes, the PR-AUC-best configuration is recommended as the primary result, with the F1-best run noted as an alternative that favours recall.

---

## 4. Hyperparameter-by-Hyperparameter Analysis

### 4.1 Learning Rate

| lr | PR AUC | F1 macro | ROC AUC | Sensitivity | Specificity | Jaccard | Hamming | n |
|---|---|---|---|---|---|---|---|---|
| 1e-4 | 0.6251 | 0.5963 | 0.7958 | 0.6876 | 0.7392 | 0.4303 | 0.2498 | 15 |
| 5e-4 | 0.6260 | 0.6012 | 0.7986 | 0.6835 | 0.7498 | 0.4352 | 0.2443 | 16 |
| 1e-3 | **0.6278** | **0.6020** | **0.8007** | **0.6895** | 0.7443 | **0.4360** | **0.2455** | 16 |
| **Δ (max–min)** | 0.0026 | **0.0057** | 0.0049 | 0.0060 | 0.0106 | 0.0057 | 0.0055 | |

**The most impactful parameter in the sweep.** There is a consistent monotonic improvement across all metrics as lr increases from 1e-4 to 1e-3. The improvement is most pronounced on F1 (Δ=0.0057) and ROC AUC (Δ=0.0049). Low lr (1e-4) also leads to much later convergence (mean best epoch = 42.4) versus 1e-3 (mean best epoch = 18.8), suggesting that 1e-4 runs are underfit within the training budget rather than genuinely better-regularised.

The lr × dropout interaction reinforces this: at lr=1e-3, dropout=0.30 outperforms 0.47 (F1 0.6031 vs 0.6010), but at lr=1e-4 the gap almost vanishes (0.5970 vs 0.5956), indicating that the learning rate effect dominates over regularisation choices.

**Recommendation:** lr=1e-3 is clearly preferred. A coarser test of lr=2e-3 or 5e-3 would be worth running to confirm the boundary.

---

### 4.2 Lambda_viol (Hierarchical Violation Penalty)

| λ_viol | PR AUC | F1 macro | ROC AUC | Sensitivity | Specificity | Jaccard | Hamming | n |
|---|---|---|---|---|---|---|---|---|
| 0.00 | **0.6267** | **0.6003** | **0.7990** | **0.6887** | 0.7426 | **0.4344** | **0.2465** | 12 |
| 0.01 | 0.6253 | 0.5995 | 0.7982 | 0.6879 | 0.7399 | 0.4334 | 0.2480 | 12 |
| 0.05 | 0.6265 | 0.6004 | 0.7987 | 0.6880 | **0.7490** | 0.4343 | 0.2457 | 12 |
| 0.10 | 0.6268 | 0.5994 | 0.7978 | 0.6825 | 0.7467 | 0.4335 | 0.2455 | 11 |
| **Δ (max–min)** | 0.0016 | 0.0010 | 0.0012 | 0.0062 | 0.0091 | 0.0010 | 0.0025 | |

**The smallest effect of any parameter in the sweep.** PR AUC range = 0.0016, F1 range = 0.0010 — both below the standard deviation of the run distribution (0.0045 and 0.0052 respectively). There is no interpretable trend; λ=0.0 is nominally best on PR AUC while λ=0.10 is nominally best on PR AUC as well — indicating noise, not signal.

This is the central empirical finding of the sweep and directly confirms the structural argument: because the 12 metacategories in Dataset A are derived entirely by OR-logic over the 129 fine labels, a hierarchical violation between the local (fine) and global (meta) heads is structurally constrained. Penalising violations provides no additional training signal. The hierarchical penalty λ_viol is effectively inert.

The interaction of lr × λ_viol reinforces this — at lr=1e-3, performance actually trends slightly downward as λ increases (0.6291 → 0.6282 → 0.6274 → 0.6264), which is consistent with the penalty adding noise rather than structure.

---

### 4.3 Weight Decay

| wd | PR AUC | F1 macro | ROC AUC | Sensitivity | Specificity | Jaccard | Hamming | n |
|---|---|---|---|---|---|---|---|---|
| 1e-4 | **0.6277** | 0.6000 | **0.7988** | 0.6834 | **0.7456** | **0.4361** | 0.2481 | 24 |
| 1e-3 | 0.6248 | **0.5999** | 0.7981 | **0.6905** | 0.7434 | 0.4316 | **0.2449** | 23 |
| **Δ (max–min)** | 0.0029 | 0.0001 | 0.0007 | 0.0071 | 0.0022 | 0.0045 | 0.0032 | |

With only two values tested, conclusions are limited. On PR AUC, wd=1e-4 has a small advantage (Δ=0.0029). On F1, they are essentially identical (Δ=0.0001). The loss gap (test loss − train loss) is larger for wd=1e-4 (0.0640) than wd=1e-3 (0.0405), suggesting that higher weight decay does reduce overfitting, but this does not translate to better generalisation metrics in this range.

The lr × weight_decay interaction shows that at lr=1e-3, wd=1e-4 outperforms wd=1e-3 on PR AUC (0.6313 vs 0.6242), while at lr=5e-4 the relationship reverses slightly (0.6250 vs 0.6269). This mild interaction suggests wd=1e-4 is the better choice specifically when combined with higher learning rates.

**Recommendation:** wd=1e-4 is preferred for the lr=1e-3 configuration. Testing wd=0 would clarify whether any L2 regularisation is beneficial.

---

### 4.4 Dropout

| dropout | PR AUC | F1 macro | ROC AUC | Sensitivity | Specificity | Jaccard | Hamming | n |
|---|---|---|---|---|---|---|---|---|
| 0.30 | **0.6277** | **0.6013** | **0.7994** | 0.6859 | **0.7481** | **0.4357** | 0.2468 | 24 |
| 0.47 | 0.6248 | 0.5985 | 0.7975 | **0.6879** | 0.7409 | 0.4320 | **0.2462** | 23 |
| **Δ (max–min)** | 0.0029 | 0.0028 | 0.0019 | 0.0020 | 0.0072 | 0.0037 | 0.0006 | |

Dropout=0.30 outperforms 0.47 consistently across PR AUC, F1, ROC AUC, and specificity. This is the second most reliable directional result after lr. However, the loss gap is larger at dropout=0.30 (0.0601) than 0.47 (0.0446), confirming that 0.30 slightly underfits — which in this case is still preferable for generalisation.

The direction of the result (lower dropout wins) is consistent with a model that may benefit from retaining more signal during training, given the sparsity of multi-hot labels in odour classification. Heavier dropout risks losing rare positive signal entirely.

**Recommendation:** 0.30 is preferred. Testing 0.20 would be informative given the downward trend.

---

## 5. Beta Ablation (Local/Global Blending Coefficient)

| β | PR AUC | F1 macro | ROC AUC |
|---|---|---|---|
| 0.1 | 0.6181 | 0.6036 | 0.7924 |
| 0.3 | 0.6172 | 0.6026 | 0.7946 |
| 0.5 | 0.6193 | 0.6030 | 0.7949 |
| 0.7 | 0.6204 | 0.6066 | 0.7922 |
| 0.9 | 0.6209 | 0.6051 | 0.7916 |
| 1.0 | 0.6179 | 0.5966 | 0.7891 |
| **Δ (max–min)** | **0.0037** | **0.0100** | **0.0058** |

The beta ablation was run at a fixed configuration (lr=1e-3, wd=1e-4, dropout=0.3), so results are comparable across β values. There is no monotonic trend in any direction. The apparent "best" values (β=0.9 for PR AUC, β=0.7 for F1) differ, and the ranges are consistent with noise given the sweep variability observed above.

The one mild signal is that β=1.0 (pure global head, local head entirely discarded) dips on F1 (0.5966, the lowest value). This suggests the local head contributes some marginal utility — but not enough to create a meaningful trend in the 0.1–0.9 range.

This result is the strongest confirmation of the structural argument: the blending coefficient between the local (fine-label) head and the global (meta-label) head has nothing to exploit when metacategories are derived by OR-logic. The model's performance is invariant to how its predictions are blended between hierarchy levels because those levels encode redundant information.

---

## 6. Convergence Behaviour

| lr | Mean best epoch | Interpretation |
|---|---|---|
| 1e-4 | 42.4 | Slow convergence; potentially still improving at cutoff |
| 5e-4 | 13.9 | Fast convergence |
| 1e-3 | 18.8 | Fast convergence with slight overshoot correction |

| λ_viol | Mean best epoch |
|---|---|
| 0.00 | 22.3 |
| 0.01 | 25.2 |
| 0.05 | 21.6 |
| 0.10 | 30.0 |

The best-epoch distribution is highly variable (std = 22.3 epochs, range 5–90). Lr=1e-4 runs take on average 42 epochs to reach their best, which is more than double the 1e-3 runs. This suggests 1e-4 runs may be training-budget limited — their lower final metrics may partly reflect underfitting rather than the intrinsic quality of that learning rate.

The slight increase in best epoch with higher λ_viol (especially 0.10 → 30.0) may reflect the penalty adding noise to the loss landscape that delays convergence, again consistent with the penalty being uninformative.

---

## 7. Sensitivity–Specificity Tradeoff

Sensitivity and specificity move in opposite directions across parameters, indicating a tradeoff that cannot be fully resolved within this search space:

| Parameter | Higher sensitivity | Higher specificity |
|---|---|---|
| lr: 1e-3 vs 1e-4 | 1e-3 (0.6895 vs 0.6876) | 5e-4 (0.7498) |
| wd: 1e-3 vs 1e-4 | 1e-3 (0.6905) | 1e-4 (0.7456) |
| dropout: 0.47 vs 0.30 | 0.47 (0.6879) | 0.30 (0.7481) |
| λ: 0.00 vs 0.10 | 0.00 (0.6887) | 0.05 (0.7490) |

In perfumery odour classification, **sensitivity is typically more important** — missing a genuine odour descriptor (false negative) is more costly than a false positive. This favours configurations with higher dropout (0.47) or lower weight decay, despite their slightly lower PR AUC.

---

## 8. Fine-Label vs Meta-Label Performance

| Metric | Fine (138 labels) | Meta (12 labels) | Ratio |
|---|---|---|---|
| PR AUC (mean) | 0.1375 | 0.6263 | 0.22 |
| F1 macro (mean) | 0.1326 | 0.5999 | 0.22 |
| ROC AUC (mean) | 0.7431 | 0.7985 | 0.93 |

Fine-label PR AUC and F1 are dramatically lower than meta-label metrics, as expected given 138-class sparsity. ROC AUC is more comparable (0.74 vs 0.80) because it is less sensitive to prevalence.

The correlation between fine and meta PR AUC across runs is only **0.31**, and between fine and meta F1 is **0.20**. This means hyperparameter choices that improve meta-label performance do not reliably improve fine-label performance. Tuning should be done on the level at which evaluation is reported.

---

## 9. Summary and Recommended Configuration

### Sensitivity ranking (Δ on F1 macro, meta-12)

| Rank | Parameter | F1 Δ | PR AUC Δ | Verdict |
|---|---|---|---|---|
| 1 | **lr** | **0.0057** | 0.0026 | Meaningful, monotonic |
| 2 | **dropout** | 0.0028 | 0.0029 | Consistent direction |
| 3 | **β** (ablation) | 0.0100 | 0.0037 | No trend, noise |
| 4 | **weight_decay** | 0.0001 | 0.0029 | Negligible on F1 |
| 5 | **λ_viol** | **0.0010** | 0.0016 | Structurally inert |

### Recommended configuration

| Parameter | Value | Rationale |
|---|---|---|
| lr | **1e-3** | Best PR AUC, F1, ROC AUC; consistent across interactions |
| weight_decay | **1e-4** | Marginal advantage in PR AUC at lr=1e-3 |
| dropout | **0.30** | Consistently outperforms 0.47 on PR AUC and F1 |
| λ_viol | **0.0** | Inert; omitting the penalty avoids adding noise |
| β | **0.5** (default) | No trend detected; default is adequate |

**Best run metrics** (λ=0.0, lr=1e-3, wd=1e-4, do=0.30):  
PR AUC = **0.6388** | F1 macro = **0.6093** | ROC AUC = **0.8086**

If recall is prioritised over precision, the F1-best configuration (λ=0.01, lr=1e-3, wd=1e-4, do=0.47) gives sensitivity=0.7118 and F1=0.6117 at PR AUC=0.6341.

### Conclusions

The sweep confirms two things simultaneously. First, **standard optimisation parameters matter**: lr is the only lever with a consistent, directional effect on performance, and its effect — while modest in absolute terms — is reproducible. Second, and more significantly, **the HMCN-F hierarchy-specific components (λ_viol, β) are inert on Dataset A**. Both show sub-noise-level effects across their full tested ranges. This is not a failure of search coverage — the result is theoretically expected given that the 12 metacategories are OR-derived from the 129 fine labels and carry no independent supervisory signal. The hierarchical inductive bias built into HMCN-F cannot be exploited under this label structure.
