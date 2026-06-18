# Project Agenda — Digital Olfaction & Multi-Label Classification

> **Internship project at ETIS** | Supervisors: Vassilis Christophides, Guillaume Renton | Colleague: Vinícius (GNN track)
>
> **Goal:** Validate that molecular structural features carry predictive signal for odor perception (Phase 1), then apply to Aryballe SPRi e-nose sensor prediction (Phase 2), then incorporate peptide sequence features (Phase 3).

---

## Table of Contents

- [Week 0 — Project Setup & Feature Engineering](#week-0--apr-28----project-setup--feature-engineering)
- [Week 1 — Open Dataset Strategy & Multi-Label Framing](#week-1--may-56----open-dataset-strategy--multi-label-framing)
- [Week 2 — JadBio Experiments & SMOTE Investigation](#week-2--may-2021----jadbio-experiments--smote-investigation)
- [Week 3 — Literature Review & Metric Decisions](#week-3--may-27----literature-review--metric-decisions)
- [Week 4 — Python Pipeline & Binary Relevance Baselines](#week-4--june-24----python-pipeline--binary-relevance-baselines)
- [Week 5 — Multi-Model Benchmark & Classifier Chains](#week-5--june-45----multi-model-benchmark--classifier-chains)
- [Week 6 — ScentTree Taxonomy Analysis](#week-6--june-9----scentree-taxonomy-analysis)
- [Week 7 — HMCN-F Implementation & Grid Search](#week-7--june-10----hmcn-f-implementation--grid-search)
- [Week 8 — HMCN-F Ablation Study & Implementation Validation](#week-8--june-1012----hmcn-f-ablation-study--implementation-validation)
- [Week 9 — Architectural Reassessment & Critical Diagnosis](#week-9--june-1718----architectural-reassessment--critical-diagnosis)
- [Key Learnings & Principles](#key-learnings--principles)
- [Current Status & Next Steps](#current-status--next-steps)

---

## Week 0 (~Apr 28) — Project Setup & Feature Engineering

### Context
Working with the Aryballe NeOse Pro SPRi e-nose device. Initial dataset: 21 rows covering 6 terpene VOCs (Ocimene, Δ3-Carene, Linalool, α-Pinene, R-Limonene, S-Limonene) with 8 peptide biosensor channels as prediction targets. Waiting for the full Aryballe dataset; task is to deeply understand the feature pipeline in advance.

### Tasks

**SMILES fetching pipeline via PubChem API**
- Built a two-tier name-cleaning system: timestamp stripping, space-to-dash conversion, and a manual `NAME_MAP` for non-standard entries (e.g. `"A-Pinene"` → `"Alpha-Pinene"`, `"Linalol"` → `"Linalool"`, `"S-limonene"` → `"(-)-Limonene"`, `"R-limonene"` → `"(+)-Limonene"`).
- Used `IsomericSMILES` (not plain `SMILES`) to preserve stereochemistry.

**R-Limonene data quality bug discovered**
- Source dataset contained incorrect SMILES for R-Limonene — a peroxide ring (`OO`) producing ascaridole instead of limonene. PubChem returned the wrong compound for the ambiguous query `"R-limonene"`.
- Fix: map to `"(+)-Limonene"` (PubChem's canonical name), correct SMILES: `CC(=C)[C@@H]1CCC(=CC1)C`.
- Consequence: all 2D descriptors were showing artifactual differences between enantiomers due to this error.

> **Key insight:** Standard 2D descriptors treat R/S enantiomers as identical (same SMILES, same graph). 3D conformer-based descriptors (PMI ratios, NPR1/NPR2 via RDKit) are required to distinguish them. This shapes the entire downstream feature strategy.

**Mordred 2D+3D descriptor pipeline (`Script_all_feature.ipynb`)**
- Features: MACCS keys (166 bits), Morgan fingerprints (radius=2, 512 bits via `MorganGenerator`), Mordred 2D+3D descriptors.
- 3D conformer generation: SMILES → add H → `EmbedMolecule(randomSeed=42)` → MMFF optimization → `RemoveHs` → `Mordred(ignore_3D=False)`.
- Three-stage Mordred filtering: drop columns with >20% NaN → median imputation for surviving NaNs → drop zero-variance → remove one from correlated pairs (|r|>0.95).
- Pipeline: 1826 raw 2D+3D descriptors → 97 after filtering.
- Added 7 visualizations: molecular structure grid, MACCS active bits heatmap, Tanimoto similarity matrix, Mordred filtering funnel, z-score normalized heatmap, sensor response heatmap, radar chart.
- Bug caught and fixed: nested double `.append()` in Morgan cell (`morgan_fps_raw.append(morgan_fps_raw.append(...))`) caused every entry to be `None`.

---

## Week 1 (~May 5–6) — Open Dataset Strategy & Multi-Label Framing

### Context
Vassilis suggested using an open-source molecule-odor dataset to validate discriminative power of molecular features while waiting for the full Aryballe dataset.

### Tasks

**Adopting the GoodScents + Leffingwell dataset**
- Downloaded `MultiLabelled_Smiles_Odors_dataset.csv`: 4,983 molecules, 138 binary odor descriptor columns (`nonStereoSMILES` + `descriptors` + 138 label columns).
- Tutor confirmed clustering the 138 descriptors into broader metacategories is the correct path.
- Tutor provided the validated `META_CATEGORIES` lookup table (12 odor families) from a previous student's work: `floral`, `fruity`, `sweet`, `woody`, `green`, `spicy`, `animal_musk`, `earthy`, `citrus`, `chemical`, `gourmand`, `powdery_amber`.
- Metacategory construction: OR-logic — a molecule is positive for a metacategory if it is positive for any of its child fine labels.

**Multi-label framing & Binary Relevance strategy agreed**
- JadBio supports only binary, multiclass, regression, and survival analysis — not native multilabel.
- Binary Relevance established as the approach: 12 independent one-vs-rest classifiers.
- 6 terpene molecules held out as a separate validation set throughout.
- 80/20 stratified split on remaining ~4,977 molecules for statistical evaluation.
- Tutor's strict NaN rule: drop any Mordred descriptor column with even one NaN — no imputation.

**`goodscents_pipeline_final.ipynb`**
- Full pipeline: MACCS + Morgan + Mordred (2D+3D) from `MultiLabelled_Smiles_Odors_dataset.csv`.
- Rows where all 3D descriptors are NaN (failed 3D embedding) dropped before the column-level NaN filter — preserves 3D features needed for enantiomer separation.
- `iterative_train_test_split` (scikit-multilearn) for multi-label stratification.

---

## Week 2 (~May 20–21) — JadBio Experiments & SMOTE Investigation

### Tasks

**JadBio AutoML — 12 one-vs-rest classifiers**
- Three labels naturally balanced (Fruity, Green, Chemical): used original data directly.
- Nine imbalanced labels: external SMOTE applied only to the training fold before passing to JadBio.
- Per-label deliverables: cumulative feature importance charts, importance-drop charts, CSV signature files.
- Macro-averaged results: ROC AUC 0.815, Balanced Accuracy 0.694.

**SMOTE vs imbalanced training — major finding**

| Label | AUC (SMOTE) | AUC (Imbalanced) |
|-------|-------------|-----------------|
| Woody | 0.520 | 0.845 |
| Animal Musk | high | low sensitivity |

- Imbalanced training consistently outperformed SMOTE across most labels.
- Animal Musk: 87.9% accuracy but only 35.4% True Positive Rate — flagged as majority-class exploitation, not genuine learning.
- Floral feature analysis: without balancing, `EState_VSA2` alone achieved ~85% — a red flag. After SMOTE, `GeomDiameter` started at ~60% with gradual multi-feature build-up, indicating genuine learning.
- Woody's near-zero original feature importance was confirmed to be a class imbalance artifact, not genuine signal absence.

> **Key principle established:** SMOTE is a statistical prior-correction tool operating in feature space, not chemical data augmentation. JadBio's internal performance metrics are unreliable for SMOTE-trained models (synthetic data contaminates internal CV folds). However, feature selection results are valid — SMOTE corrects majority-class bias in Boruta.

**JadBio type mismatch debugging**
- Root cause: SMOTE converts binary fingerprint columns (int 0/1) to floats (0.0/1.0). JadBio auto-infers column types from CSV values — training data inferred as continuous, test data as categorical → type mismatch error cascading column by column (`MACCS_126`, then `morgan_409`, etc.).
- External dtype matching was insufficient because JadBio applies its own internal type inference independently of Python dtypes.
- Solution: download training data directly from JadBio's export function (reflects JadBio's internal type assignments) and use it as the reference for aligning the test file.

**Analysis report + 26-slide PowerPoint presentation**
- Formal Word report: per-label feature selection analyses, cross-label synthesis.
- Recurring important features across labels: `MACCS_123`, `MACCS_154`, `piPC` family, `nS`, `SLogP`.
- 26 slides appended to existing presentation deck (dark navy/white/orange style, built with python-pptx + matplotlib). Speaker notes markdown file generated separately.

---

## Week 3 (~May 27) — Literature Review & Metric Decisions

### Tasks

**Paper review: Wen et al. 2025 (*Molecules*)**
- GNN vs traditional QSAR on 3,304 GoodScents molecules, 6 odor labels.
- Key finding: decision threshold optimization for imbalanced multi-label classification.
- Framework: PyTorch Geometric + standard scikit-learn (not scikit-multilearn).
- Directly relevant to the GNN track (Vinícius) and threshold optimization strategy.

**Paper review: Ameta et al. 2025 (*PLOS ONE*)**
- Compared molecular fingerprints, vibrational spectra, and mass spectra across multi-label strategies on a 7,374-molecule integrated dataset with 109 odor classes.
- BR vs CC vs RF across Mordred/Morgan/Daylight features.
- Uses scikit-multilearn for iterative stratified splits and ML-ROS/ML-RUS resampling (Szymański & Kajdanowicz 2017).
- Informed multi-label method selection and alternatives to SMOTE.

**Label co-occurrence matrix analysis**
- Studied the 109×109 co-occurrence matrix approach for measuring label dependency.
- Understood Louvain community detection for label clustering.
- MeanIR and IRLbl metrics for quantifying label imbalance.

**Macro-averaged F1 confirmed as tuning metric**
- Computes F1 independently per label before averaging — treats all labels equally regardless of prevalence.
- Formally agreed with Vassilis via email.
- PR AUC confirmed as the preferred reporting metric given label imbalance.
- MCC dropped from the agreed metric set.

---

## Week 4 (~June 2–4) — Python Pipeline & Binary Relevance Baselines

### Tasks

**Full preprocessing pipeline in Python (`baseline.ipynb`)**

Rebuilt independently of JadBio from `goodscents_jadbio_ready.csv` (4,981 molecules, semicolon separator):

| Step | Action | Detail |
|------|--------|--------|
| 1 | NaN inspection | 5 rows affected (all Mordred) → drop rows (0.1% data loss) |
| 2 | Feature separation | MACCS + Morgan (binary, no scaling) vs Mordred (continuous) |
| 3 | Zero-variance removal | 6 MACCS bits dropped via `VarianceThreshold` |
| 4 | Stratified split 80/20 | `iterative_train_test_split` — 3,877 train / 1,099 test |
| 5 | StandardScaler on Mordred | Fitted on train only; applied to test to avoid data leakage |
| 6 | Correlation filter | Drop one of each pair with Pearson \|r\| > 0.95 (train only) |

Final: **999 features**, 4,976 rows.

**BR + Logistic Regression baseline**
- `BinaryRelevance` wrapper (scikit-multilearn) + LR with `class_weight='balanced'`, `solver='saga'`.
- `GridSearchCV` over C ∈ {0.01, 0.1, 1, 10, 100}, 3-fold CV scored on macro-F1.
- Results: Macro Balanced Accuracy 0.741, Macro Sensitivity 0.756.
- BR+LR outperformed JadBio on sensitivity (0.756 vs 0.468). JadBio won on PR AUC (0.779 vs 0.576) and Specificity.
- Citrus (10.3% prevalence) confirmed as the hardest label across all methods.

> **Key pattern:** BR+LR leads on sensitivity and balanced accuracy. JadBio and BR+RF both trade sensitivity for specificity. High ROC AUC with near-zero sensitivity signals majority-class exploitation, not genuine learning.

**BR + Random Forest added**
- `GridSearchCV` over `n_estimators` ∈ {100, 200, 300}, `max_features` ∈ {sqrt, log2}.
- Results: Bal.Acc=0.689, MCC=0.432, F1=0.546, ROC AUC=0.820, PR AUC=0.624, Sensitivity=0.487, Specificity=0.892.

**TikZ preprocessing diagram (Beamer)**
- Two-slide LaTeX Beamer diagram. Ghost-step continuity anchor at top of slide 2. "1/2" progress indicator in frame title.
- Raw features (1,344) → 999 final features visualized step by step.

**Thursday presentation to supervisors**
- 8-slide Beamer: Project Overview → Dataset & Label Taxonomy → Molecular Features → Preprocessing Pipeline → Models & Tuning → Evaluation Metrics → Results (BR+LR vs JadBio) → Next Steps.

---

## Week 5 (~June 4–5) — Multi-Model Benchmark & Classifier Chains

### Tasks

**5-model unified BR benchmark**
- Loop over: Logistic Regression, Random Forest, XGBoost, SVM (RBF kernel), KNN — all wrapped in `BinaryRelevance`.
- Shared `evaluate()` and `print_results()` helpers. Summary macro-average table at the end.

**Classifier Chains & ECC (`classifier_chains.ipynb`)**
- CC with RF base estimator via sklearn's `ClassifierChain`.
- ECC: 10 chains with different random seeds, majority vote for label predictions, averaged probabilities for PR AUC.
- Default label ordering: by frequency (most frequent first).

**Label Powerset rejected**
- With 12 labels: 2^12 = 4,096 possible label combinations. Most combinations have essentially no examples in ~5K samples.
- LP treats each combination as a separate atomic class — similar combinations share no information despite sharing labels.
- PyCaret also assessed and rejected: no native multilabel support, incompatible with custom preprocessing pipeline.

**GA-optimized label ordering for CC (pending)**
- Reviewed a paper on genetic algorithm-optimized label ordering: bi-level optimization, ExF-cor fitness function (example-based F-measure + coverage ratio), tournament selection, order crossover (OX), swap mutation.
- Decision: run BR vs CC vs ECC comparison first before implementing GA ordering search.

---

## Week 6 (~June 9) — ScentTree Taxonomy Analysis

### Context
Vassilis recommended grounding the metacategory dictionary in ScentTree — an expert-curated formal olfactory taxonomy — rather than relying solely on the previous student's hand-crafted lookup table.

### Tasks

**ScentTree JSON structural analysis**
- ScentTree is a **3-layer DAG** (directed acyclic graph), not a strict tree.
- Stats: 17 Layer 1 root nodes, 101 total unique node names across all layers.
- DAG property: nodes can have multiple parents. Example: "black currant" connects to both Berries and Sulfuric at Layer 1.
- Critical clarification: the DAG structure affects only the label construction step (OR-logic setting multiple metacategory bits to 1). It does not affect the internal architecture of any downstream model.

**Mapping the 138 fine labels**

| Category | Count | Notes |
|----------|-------|-------|
| Direct ScentTree matches | 63 | Matched by name or synonym |
| Absent from ScentTree | 75 | Not a data gap — ScentTree uses abstract family names |

ScentTree's design uses abstract family names (e.g. "Yellow Fruits", "Cut Grass") while the dataset uses specific descriptors (e.g. "apple", "hay"). This is a deliberate taxonomic design choice, not an incomplete file.

**All-paths vs 2-hop mapping strategies**

| Strategy | Avg Layer 1 parents per label | Notes |
|----------|-------------------------------|-------|
| All-paths | 3.83 | Includes weak distant associations |
| 2-hop | 2.86 | Filters weak links — **recommended** |

**Layer 1 assignments for all 75 absent labels**

Proposed assignments with confidence ratings (Strong / Medium / Weak / Ambiguous). Examples:

| Fine label | Proposed Layer 1 | Confidence |
|-----------|-----------------|------------|
| apple | Fruity | Strong |
| black currant | Fruity, Sulfuric | Strong |
| hay | Green, Herbal, Undergrowth | Strong |
| ethereal | Solvents, Fruity | Strong |
| chamomile | Herbal, Floral | Strong |
| natural | — | Ambiguous |
| odorless | Drop | Ambiguous |

Delivered as `scentree_layer1_assignments.csv` (using `|` as separator for multi-parent fields to avoid CSV ambiguity).

**Deliverables**
- Word report: DAG vs tree clarification, 63/75 split analysis, all-paths vs 2-hop comparison, full assignment table with confidence, recommendations for supervisor meeting.
- Jupyter notebook: replicates the full ScentTree analysis with DAG structure visualizations and mapping statistics.
- Status: **pending Vassilis review** before metacategory dictionary is updated in the pipeline.

---

## Week 7 (~June 10) — HMCN-F Implementation & Grid Search

### Tasks

**`hmcn_dataset.csv` construction (`build_hmcn_dataset.py`)**
- Merged molecular features from `goodscents_jadbio_ready.csv` with 138 fine labels + 12 meta labels from `MultiLabelled_Smiles_Odors_dataset.csv`.
- Output: `hmcn_dataset.csv` — 4,976 rows × 1,156 columns (SMILES + 166 MACCS + 512 Morgan + ~478 Mordred + 138 fine labels + 12 meta labels).

**HMCN-F Option A — full PyTorch implementation (`hmcn_final_final_final.ipynb`)**

Architecture (two-level hierarchy):
- Level 1 (local): 138 fine odor labels
- Level 0 (global): 12 metacategories
- Global branch: shared MLP backbone producing global predictions P_G (dim=12)
- Local branches: one `LocalBlock` per hierarchy level producing P_L (dim=138)
- **Eq 6 combination:** P_F = β · P_L + (1−β) · P_G
- **Hierarchical violation penalty** (Equations 13–17): penalizes P(child) > P(parent)
- Optimizer: Adam, `weight_decay=1e-4`
- Stratification: `MultilabelStratifiedShuffleSplit` for train/val/test
- Tuning criterion: validation Meta ROC AUC

**24-config hyperparameter grid search (Google Colab GPU)**

| Hyperparameter | Values searched |
|---------------|----------------|
| Learning rate | {1e-3, 1e-4} |
| Dropout | {0.3, 0.47} |
| global_dim | {128, 256} |
| lambda_viol (λ) | {0.05, 0.1, 0.5} |
| beta (β) | {0.3, 0.5, 0.7} |

- Scheduler: `ReduceLROnPlateau`
- Best config (selected by val Meta ROC AUC, tiebroken by test PR AUC): lr=1e-4, dropout=0.47, weight_decay=1e-4
- **HMCN-F did not outperform BR baselines** → Vassilis requested ablation study.

**Focused 6-config re-tuning**
- Switched scheduler to `CosineAnnealingWarmRestarts`.
- Added per-label threshold optimization (search per label, not fixed 0.5).
- Added validation loss tracking to training loop.
- Removed LR schedule plot from training curve visualization.
- Still did not beat BR baselines.

**BR notebook updated + joint Beamer presentation with Vinícius**
- Added validation split, per-label threshold search, instance-based metrics.
- Fixed cell ordering bug: threshold search cell must precede evaluation cell.
- Merged 7 slides into Vinícius's existing GNN presentation at correct logical positions.
- PR AUC set as primary reporting metric. MCC formally dropped.

---

## Week 8 (~June 10–12) — HMCN-F Ablation Study & Implementation Validation

### Tasks

**Systematic ablation — 3 parameter dimensions**

Three separate notebooks, each varying one parameter:

| Notebook | Parameter | Values | Key question |
|----------|-----------|--------|-------------|
| `hmcn_lambda_ablation.ipynb` | lambda_viol | {0.0, 0.05, 0.1, 0.5, 1.0} | Does the hierarchy penalty help or hurt? |
| `hmcn_capacity_ablation.ipynb` | global_dim × local_dim | {64,128,256} × {32,64,128} | Overfitting vs underfitting? |
| `hmcn_beta_ablation.ipynb` | beta | {0.2, 0.3, 0.5, 0.7, 0.8} | How much to trust local vs global predictions? |

- Shared `hmcn_eval.py` module: identical metric definitions across all experiments (prevents inconsistency).
- All results saved to `hmcn_ablation_results.csv`.
- Results location in Colab: `/content/hmcn_ablation_results.csv` (recommend saving directly to Google Drive).

**eisen_GO sanity check (`hmcn_sanity_check_eisen_GO.ipynb`)**

Goal: separate implementation correctness from architectural fitness.

- Reproduces paper Table 4 results on the original eisen_GO dataset (CLUS benchmark).
- Target: AU(PRC) ≈ 0.440.
- 6 hierarchy depth buckets (N_LEVELS=6), one LocalBlock per bucket — matches the paper's multi-level structure.
- Pass criterion: AU(PRC) ∈ [0.39, 0.49] → confirms LocalBlock, Eq 6, loss Equations 13–17, and violation penalty are all correctly implemented.
- This validation is independent of whether HMCN-F is the right architectural choice.

**Weight decay clarification**
- `weight_decay=1e-4` is L2 regularization embedded in the Adam optimizer: penalty term λθ added to gradient, applying to both global and local HMCN-F branches.
- Distinction: standard Adam+L2 vs AdamW (decoupled weight decay). Standard Adam is appropriate and conventional for this task.

---

## Week 9 (~June 17–18) — Architectural Reassessment & Critical Diagnosis

### Tasks

**HMCN-F architectural mismatch formally diagnosed**

A colleague raised a concern about the Eq 6 dimension mismatch. Deep analysis of the original paper's benchmark datasets revealed the root problem:

| | Original HMCN-F setting (FunCat/GO) | This project |
|---|---|---|
| Label source | Independent expert annotation at every level | OR-logic over child labels |
| Parent node information | Genuine independent signal absent from fine labels | Zero independent signal — fully determined by children |
| Coarse-level annotation | A protein can be labeled at FunCat level with no fine label | Impossible by construction |
| Eq 6 applicability | P_G and P_L have aligned semantics | Dimension mismatch; semantics don't align |

> **Conclusion:** HMCN-F's underperformance is not a tuning problem. It is a fundamental architectural mismatch. The model's core premise — that parent nodes carry independent information — does not hold for OR-derived synthetic metacategories.

> **Generalizable lesson:** Match model assumptions to dataset construction method, not just surface-level task framing. An OR-derived hierarchy is not the same as a hierarchy built from independent expert annotations at each level.

**Alternative method landscape assessed**

Rejected for this scale (~5K samples, 12 labels):
- SLEEC, LEML — require tens of thousands of samples.

Appropriate alternatives:
- ML-KNN
- LIFT
- RAkEL
- PLST / CPLST
- Plain MLP with BCE loss
- **BR+LR and BR+RF (already implemented) — among the most appropriate methods for this actual problem setup.**

**ScentTree Layer 1 CSV finalized**
- `scentree_layer1_assignments.csv` delivered with `|` separator for multi-parent fields.
- Status: pending Vassilis review before metacategory dictionary replacement is implemented.

---

## Key Learnings & Principles

### HMCN-F architectural mismatch
HMCN-F was designed for biological ontologies (FunCat, Gene Ontology) where every hierarchy node is independently expert-annotated. Synthetically derived (OR-logic) metacategories provide no independent signal at the coarse level, undermining the core HMCN-F premise. **Match model assumptions to dataset construction method, not just surface-level task framing.**

### Scale-appropriate model selection
Methods like SLEEC and LEML require far larger datasets. For ~5,000 samples with 12 labels, BR baselines and ensemble methods are competitive and appropriate. Complexity should be justified by the data structure, not the problem's perceived sophistication.

### SMOTE limitations in cheminformatics
SMOTE applied to binary fingerprint columns (integer 0/1) produces float values, creating type inference mismatches in platforms like JadBio. Imbalanced training consistently outperformed SMOTE for most labels — most dramatically Woody (AUC 0.520 with SMOTE vs 0.845 without). SMOTE is a statistical prior-correction tool operating in feature space, not a chemical data augmentation method.

### Evaluation metric discipline
- **Tuning metric:** Macro-averaged F1 (treats all labels equally regardless of imbalance).
- **Reporting metrics:** PR AUC (preferred given imbalance) and ROC AUC.
- **Dropped:** MCC.
- High ROC AUC with near-zero sensitivity (e.g. Animal Musk) signals a model exploiting majority-class structure, not genuine learning.

### Feature selection validity under SMOTE
JadBio's internal performance metrics from SMOTE-augmented training are unreliable (synthetic data contaminates internal CV folds), but feature selection results are retained as valid — SMOTE corrects majority-class bias in Boruta.

### Enantiomer distinction requires 3D descriptors
Standard 2D descriptors treat R-Limonene and S-Limonene identically. 3D conformer-based descriptors (PMI ratios, NPR1/NPR2) are necessary to distinguish them.

### R-Limonene SMILES data quality
The source dataset's R-Limonene SMILES contained an incorrect peroxide group — a data quality issue that caused artifactual feature differences across all descriptor families.

---

## Current Status & Next Steps

### Completed (Phase 1 — Validation)
- [x] Feature engineering pipeline: MACCS + Morgan + Mordred (2D+3D)
- [x] JadBio AutoML baseline across 12 labels
- [x] Python BR baseline: BR+LR, BR+RF, BR+XGBoost, BR+SVM, BR+KNN
- [x] Classifier Chains (CC) and Ensemble of Classifier Chains (ECC)
- [x] HMCN-F implementation, 24-config grid search, focused re-tuning, ablation study
- [x] ScentTree taxonomy analysis — 138 labels mapped, assignments proposed

### Pending decisions
- [ ] Vassilis review of ScentTree Layer 1 assignments → finalize metacategory dictionary replacement
- [ ] BR vs CC vs ECC comparison results → decide on GA label ordering implementation
- [ ] Decision on whether to abandon HMCN-F in favor of scale-appropriate alternatives (ML-KNN, LIFT, RAkEL, plain MLP with BCE)

### Up next (Phase 2 — E-Nose Prediction)
- [ ] Apply validated molecular features to the Aryballe dataset
- [ ] Predict SPRi sensor channel responses from molecular structure
- [ ] Distinguish R/S-Limonene enantiomers using 3D descriptors

### Up next (Phase 3 — Peptide Features)
- [ ] Incorporate peptide sequence features from Aryballe sensor chip
- [ ] Find correlations between odor perception labels and e-nose sensor outputs

---

## Key Files

| File | Description |
|------|-------------|
| `Script_all_feature.ipynb` | Feature engineering: MACCS + Morgan + Mordred 2D+3D for 6 terpenes |
| `goodscents_pipeline_final.ipynb` | Full pipeline on GoodScents + Leffingwell dataset |
| `baseline.ipynb` | Binary Relevance baselines (LR, RF, XGBoost, SVM, KNN) |
| `classifier_chains.ipynb` | Classifier Chains and ECC |
| `build_hmcn_dataset.py` | Merges features + fine labels + meta labels → hmcn_dataset.csv |
| `hmcn_final_final_final.ipynb` | HMCN-F implementation + grid search + evaluation |
| `hmcn_focused_tune.ipynb` | Focused 6-config re-tuning with CosineAnnealingWarmRestarts |
| `hmcn_tune.py` | Grid search script for Colab GPU |
| `goodscents_jadbio_ready.csv` | 4,981 molecules × features + 12 metacategory labels (semicolon delimited) |
| `MultiLabelled_Smiles_Odors_dataset.csv` | 4,983 molecules × 138 fine labels |
| `hmcn_dataset.csv` | 4,976 molecules × 1,156 columns (features + fine + meta labels) |

## Key References

- Read et al. 2011 — Classifier Chains
- Zhang & Zhou 2014 — Binary Relevance
- Wen et al. 2025 — GNN vs QSAR on GoodScents
- Ameta et al. 2025 — Multi-label strategies, scikit-multilearn
- Chacko et al. 2020 — Mordred for odor
- Orosz et al. 2022 — Descriptor benchmarking
- Debnath et al. 2023 — Mordred ≈ Morgan on odor tasks
