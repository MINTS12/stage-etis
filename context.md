# Digital Olfaction & Machine Learning — Project README

**Intern:** Nahla  
**Supervisors:** Guillaume, Vassilis  


---

## 1. Project Overview

The goal of this project is to build a machine learning pipeline that **predicts sensor responses of an electronic nose (e-nose) from molecular structure alone**, without needing to physically measure the molecule.

The device is the **Aryballe NeOse Pro** — an opto-electronic nose based on Surface Plasmon Resonance imaging (SPRi) with a peptide biosensor array. It produces an 8-channel sensor response vector for each volatile organic compound (VOC) it measures.

The project has **two parallel tracks:**

### Track 1 — Validation (Olfactory Multi-label Classification)
Before applying molecular features to predict e-nose signals, we validate that our chosen features (MACCS, Morgan, Mordred) carry genuine olfactory predictive signal. We do this on a large external dataset (~5K molecules) where ground truth odor labels are known.

### Track 2 — E-nose Prediction
The core objective: predict the 8 SPRi sensor channel values of a molecule from its molecular structure. Currently working with 6 terpenes measured in triplicate.

---

## 2. Hardware & Data — E-nose Track

**Device:** Aryballe NeOse Pro (SPRi peptide biosensor array)  
**Sensor channels:** 8 (columns: `1`, `105`, `106`, `24`, `25`, `34`, `36`, `55`)  
**Molecules measured:** 6 terpenes, each in triplicate (18 measurements total)

| Molecule | Notes |
|---|---|
| Ocimene | |
| Δ3-Carene | |
| Linalool | |
| α-Pinene | |
| R-Limonene | ⚠ SMILES in source file may contain peroxide group — verify against PubChem CID 440491 |
| S-Limonene | |

**Source file:** `7Q27_signatures_with_smiles.csv` (semicolon-delimited)  
**Normalized signatures:** `7Q27_normalized_signatures.csv`  
**Full feature dataset:** `dataset_all_features_plots.csv`

---

## 3. Molecular Feature Pipeline

Three complementary feature families are computed from SMILES strings via RDKit and Mordred. They are complementary by design — each captures a different aspect of VOC-peptide binding physics.

### 3.1 MACCS Keys (166 bits)
- Binary structural checklist — presence/absence of predefined substructures
- Fixed, interpretable, no NaN for standard molecules
- Computed via RDKit

### 3.2 Morgan Fingerprints (512 bits, radius=2)
- Local chemical neighborhood hashing
- Encodes circular substructure environments around each atom
- Binary, no NaN
- Note: chirality awareness is insufficient to fully separate enantiomers like R- and S-Limonene (different scent: orange vs lemon) — they differ only at a chiral carbon
- When dataset scales up: increase to `nBits=2048`

### 3.3 Mordred Descriptors (~327 after filtering)
- Quantitative physicochemical properties (molecular weight, topological indices, surface areas, etc.)
- Continuous values on very different scales → requires StandardScaler
- Filtering strategy:
  1. Drop features with >20% NaN (none in current dataset — all have exactly 5 NaN rows)
  2. Drop the 5 problematic rows (Mordred computation failures on unusual/large molecules)
  3. Remove zero-variance features
  4. Remove one feature from each highly correlated pair (r > 0.95)

**SMILES retrieval:** PubChem API → SMILES → RDKit → features

---

## 4. Validation Track — Multi-label Odor Classification

### 4.1 Dataset
- **Source:** GoodScents + Leffingwell PMP 2001 compendium
- **Size:** 4,981 molecules (4,976 after dropping 5 Mordred-invalid rows)
- **File:** `goodscents_jadbio_ready.csv` (semicolon-delimited)

### 4.2 Label Reduction: 138 → 12 Metacategories
Raw odor descriptors were reduced from 138 to 12 metacategories using a pre-established lookup table from a previous student. The motivation is dual:
1. **Data sparsity** — with 138 labels most have very few positive examples
2. **Platform constraint** — JadBio cannot handle multi-label natively; 138 one-vs-rest models would be impractical

| Metacategory | Prevalence |
|---|---|
| floral | 26.2% |
| fruity | 45.8% |
| sweet | 38.7% |
| woody | 21.2% |
| green | 45.2% |
| spicy | 21.6% |
| animal_musk | 15.5% |
| earthy | 20.2% |
| citrus | 10.3% |
| chemical | 43.1% |
| gourmand | 24.2% |
| powdery_amber | 20.6% |

All labels have <50% prevalence → class imbalance must be addressed.

### 4.3 Class Imbalance Strategy

**SMOTE** (Synthetic Minority Oversampling Technique) was applied selectively to imbalanced labels in JadBio runs.

Key methodological point: **SMOTE is valid for feature selection but not for final performance evaluation.**
- SMOTE corrects statistical priors → feature selection works on a balanced distribution
- SMOTE evaluation metrics are discarded; only feature selection results are retained
- This is the "statistical prior correction" argument, not chemical augmentation

For scikit-learn models: class imbalance is handled via `class_weight='balanced'` in Logistic Regression — a cost-sensitive approach that penalizes misclassification of minority class samples more heavily:

$$w_{positive} = \frac{n_{samples}}{2 \times n_{positive}}$$

### 4.4 Preprocessing Pipeline (for scikit-learn models)

1. **Load data** — 4,981 molecules, 12 labels, 1,005 features
2. **Inspect NaN** — 5 rows affected, all in Mordred (327/327 Mordred columns), 0 in fingerprints
3. **Drop 5 problematic rows** — Mordred computation failures, 0.1% data loss → 4,976 molecules remain
4. **Split features into two groups:**
   - Fingerprints: MACCS + Morgan (678 features, binary, no scaling needed)
   - Mordred: 327 continuous features, requires StandardScaler
5. **Remove zero-variance features** — applied independently to each group (6 MACCS bits dropped, 0 Mordred)
6. **Stratified train/test split** — `iterative_train_test_split` from scikit-multilearn (80/20) preserves each label's positive/negative ratio in both sets
7. **Scale Mordred** — `StandardScaler` fitted on train only, applied to both train and test (no data leakage)
8. **Correlation filter on Mordred** — computed on train only; for each pair with r > 0.95, one feature dropped; same column mask applied to test
9. **Concatenate** — final `X_train` and `X_test` = fingerprints + filtered scaled Mordred

### 4.5 Models

#### Baseline 1 — Majority Class Classifier
Always predicts the majority class (0) for every label. Since all labels have <50% prevalence, this means always predicting "absent." Results in F1 = 0 for all labels — the absolute floor.

#### Baseline 2 — Binary Relevance + Logistic Regression (BR+LR)
- **Framework:** scikit-multilearn `BinaryRelevance` wrapping sklearn `LogisticRegression`
- **Imbalance:** `class_weight='balanced'`
- **Solver:** `saga` (efficient for large feature spaces)
- **Tuning:** `GridSearchCV` over `classifier__C ∈ {0.01, 0.1, 1, 10, 100}`, 3-fold CV, macro-F1 scorer

The math: Binary Relevance trains 12 independent classifiers. For each label $l$:

$$P(y_l = 1 | \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}_l^T \mathbf{x} + b_l)}}$$

Weights minimized via regularized cross-entropy:

$$\mathcal{L}(\mathbf{w}_l) = -\sum_{i=1}^{n} \left[ y_l^{(i)} \log P + (1-y_l^{(i)}) \log(1-P) \right] + \frac{1}{C}\|\mathbf{w}_l\|^2$$

GridSearchCV finds optimal C:

$$C^* = \arg\max_{C \in \mathcal{G}} \frac{1}{3}\sum_{k=1}^{3} \frac{1}{12}\sum_{l=1}^{12} F1_l^{(k)}(C)$$

Total fits: 5 C values × 3 folds × 12 labels = **180 logistic regressions**

#### JadBio AutoML
- Binary Relevance (one-vs-rest), Aggressive Feature Selection mode
- Internal cross-validation for hyperparameter tuning
- Outputs: Signatures, Feature Importance, Progressive Feature Importance

### 4.6 Evaluation Metrics

All metrics computed per label then macro-averaged:

| Metric | Formula | Why |
|---|---|---|
| Balanced Accuracy | (Sensitivity + Specificity) / 2 | Corrects for imbalance |
| MCC | (TP·TN - FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Robust to imbalance, uses all confusion matrix cells |
| F1 | 2·Precision·Recall / (Precision+Recall) | Optimization metric for GridSearchCV |
| ROC AUC | Area under ROC curve | Threshold-free, robust to imbalance |
| PR AUC | Area under Precision-Recall curve | Focuses on positive class, best for rare labels |
| Sensitivity | TP / (TP + FN) | True positive rate |
| Specificity | TN / (TN + FP) | True negative rate |

**Macro-average** is used throughout — each label contributes equally regardless of prevalence.

### 4.7 Results

#### BR+LR Results

| Label | Bal.Acc | MCC | F1 | ROC AUC | PR AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| floral | 0.812 | 0.546 | 0.656 | 0.875 | 0.672 | 0.854 | 0.771 |
| fruity | 0.772 | 0.537 | 0.736 | 0.852 | 0.804 | 0.776 | 0.768 |
| sweet | 0.682 | 0.348 | 0.609 | 0.742 | 0.595 | 0.784 | 0.579 |
| woody | 0.771 | 0.454 | 0.564 | 0.837 | 0.567 | 0.763 | 0.779 |
| green | 0.713 | 0.419 | 0.673 | 0.776 | 0.667 | 0.740 | 0.686 |
| spicy | 0.737 | 0.380 | 0.507 | 0.805 | 0.505 | 0.795 | 0.678 |
| animal_musk | 0.703 | 0.302 | 0.406 | 0.797 | 0.442 | 0.662 | 0.743 |
| earthy | 0.738 | 0.395 | 0.515 | 0.820 | 0.571 | 0.700 | 0.776 |
| citrus | 0.766 | 0.354 | 0.393 | 0.851 | 0.390 | 0.735 | 0.797 |
| chemical | 0.717 | 0.423 | 0.668 | 0.779 | 0.654 | 0.765 | 0.670 |
| gourmand | 0.748 | 0.419 | 0.557 | 0.817 | 0.584 | 0.776 | 0.719 |
| powdery_amber | 0.739 | 0.393 | 0.515 | 0.810 | 0.465 | 0.727 | 0.751 |
| **MACRO AVG** | **0.741** | **0.414** | **0.567** | **0.813** | **0.576** | **0.756** | **0.726** |

#### JadBio Results

| Label | Bal.Acc | MCC | F1 | ROC AUC | PR AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| floral | 0.750 | 0.526 | 0.638 | 0.857 | 0.817 | 0.595 | 0.905 |
| fruity | 0.799 | 0.600 | 0.778 | 0.872 | 0.871 | 0.762 | 0.835 |
| sweet | 0.681 | 0.366 | 0.603 | 0.753 | 0.749 | 0.587 | 0.775 |
| woody | 0.691 | 0.468 | 0.530 | 0.845 | 0.800 | 0.427 | 0.955 |
| green | 0.709 | 0.426 | 0.665 | 0.796 | 0.796 | 0.623 | 0.794 |
| spicy | 0.629 | 0.354 | 0.410 | 0.794 | 0.740 | 0.302 | 0.956 |
| animal_musk | 0.665 | 0.449 | 0.472 | 0.805 | 0.753 | 0.354 | 0.975 |
| earthy | 0.683 | 0.470 | 0.516 | 0.802 | 0.776 | 0.401 | 0.964 |
| citrus | — | — | — | — | — | — | — |
| chemical | 0.728 | 0.466 | 0.678 | 0.800 | 0.787 | 0.639 | 0.817 |
| gourmand | 0.671 | 0.450 | 0.502 | 0.845 | 0.807 | 0.377 | 0.964 |
| powdery_amber | 0.628 | 0.317 | 0.403 | 0.792 | 0.717 | 0.323 | 0.933 |
| **MACRO AVG** | **0.694** | **0.445** | **0.563** | **0.815** | **0.783** | **0.490** | **0.898** |

#### Key observations

- **BR+LR beats JadBio on:** Balanced Accuracy (0.741 vs 0.694), Sensitivity (0.756 vs 0.490) → catches more true positives
- **JadBio beats BR+LR on:** PR AUC (0.783 vs 0.576), Specificity (0.898 vs 0.726) → more precise, fewer false positives
- **ROC AUC comparable:** BR+LR 0.813 vs JadBio 0.815 — essentially the same ranking ability
- The trade-off is interpretable: BR+LR with `class_weight='balanced'` is tuned to find positives; JadBio is more conservative

---

## 5. Key Files

| File | Description |
|---|---|
| `goodscents_jadbio_ready.csv` | Full validation dataset: 4,981 molecules × 12 labels + 1,005 features |
| `7Q27_normalized_signatures.csv` | E-nose normalized sensor responses for 6 terpenes |
| `7Q27_sensograms.csv` | Raw sensogram time series |
| `dataset_all_features_plots.csv` | E-nose dataset with all molecular features |
| `preprocessing_pipeline.ipynb` | Full preprocessing + BR+LR model notebook |
| `baseline_multilabel.ipynb` | Majority class baseline + comparison table |

---

## 6. Key Methodological Decisions & Justifications

| Decision | Justification |
|---|---|
| 138 → 12 label reduction | Data sparsity + JadBio platform constraint |
| SMOTE for feature selection only | Corrects statistical priors; discarded for performance metrics |
| MACCS + Morgan + Mordred | Complementary: structural checklist + neighborhood hashing + physicochemical |
| Mordred scaling (StandardScaler) | Continuous descriptors on different scales; LR is scale-sensitive |
| No scaling for fingerprints | Binary features; scaling changes nothing |
| Correlation filter on Mordred only | Fingerprint bits encode structurally distinct substructures; Mordred has redundant descriptors |
| iterative_train_test_split | Multi-label stratification preserves label ratios in both sets |
| class_weight='balanced' | Cost-sensitive imbalance handling without oversampling |
| macro-F1 as GridSearchCV scorer | Equal weight per label regardless of prevalence |
| saga solver | Efficient for large feature spaces (~1,000 features) |
| 3-fold CV | Balance between reliability and computation time |

---

## 7. Next Steps

### Immediate 
- Run remaining JadBio label analyses (citrus missing from current export)
- Compare BR+LR vs JadBio on all metrics in the presentation

### Short term — Improve multi-label models
- **BR + XGBoost / Gradient Boosting** — motivated by Chacko et al. 2020 (Sci. Rep.) who found XGBoost optimal for odor classification; tuned with GridSearchCV
- **BR + Random Forest / SVM** — additional baselines for comparison
- **Classifier Chains** — models label dependencies (e.g. if floral → more likely sweet); addresses main limitation of BR
- **Label Powerset** — treats each unique label combination as a class; exact multi-label modeling
- All tuned with GridSearchCV + macro-F1 scorer

### Medium term — E-nose prediction track
- Apply validated molecular features to predict SPRi sensor channel values from molecular structure
- Regression problem: input = molecular features, output = 8 sensor channel values
- Verify R-Limonene SMILES (potential peroxide group issue — check PubChem CID 440491)

### Long term — Scale up
- When larger, more diverse dataset arrives:
  - Scale Morgan fingerprints to `nBits=2048`
  - Consider transition to true multi-label neural networks with shared representation layers
  - Consider Classifier Chains or Label Powerset at scale

---

## 8. Literature

| Paper | Relevance |
|---|---|
| Chacko et al. 2020 (Sci. Rep.) | Mordred for odor prediction; XGBoost + Gradient Boosting as best classifiers |
| Orosz et al. 2022 (Front. Chem.) | Mordred and MACCS feature ranking for odor tasks |
| Debnath et al. 2023 (PLoS ONE) | Mordred ≈ Morgan performance on odor classification |
| Slimani et al. 2020 (Chemosensors) | NeOse Pro + silicon micro pre-concentrator for flavored water analysis |
