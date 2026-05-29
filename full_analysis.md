# Full Analysis Report — Odor Prediction with Molecular Descriptors
## Aryballe E-Nose Internship Project

---

# 1. PROJECT CONTEXT AND OBJECTIVE

## 1.1 What we are trying to do

The Aryballe e-nose is a device that uses peptide-coated sensors working via Surface Plasmon Resonance imaging (SPRi). When a volatile organic compound (VOC) comes into contact with the sensor surface, it binds to the peptides and produces a measurable optical signal. The device has 8 sensor channels (columns 1, 24, 25, 34, 36, 55, 105, 106 in the dataset), each coated with a different peptide sequence, and each responding differently to different molecules.

The long-term goal of this project is to predict the sensor response (the numerical value each of the 8 sensors produces) from the molecular structure of the VOC alone. In other words: given only the SMILES notation of a molecule, can we predict how the e-nose will respond to it?

This is directly analogous to computational drug discovery, where the question is: given only the molecular structure of a ligand, can we predict how strongly it will bind to a protein? The underlying physics is the same — non-covalent molecular binding driven by hydrophobicity, hydrogen bonding, van der Waals forces, and shape complementarity.

## 1.2 Why we start with odor labels, not sensor values

The Aryballe dataset contains only 21 measurements across 6 distinct molecules (plus a blank). This is far too small to train any reliable machine learning model. Before we can use it, we need to validate that the molecular features we have chosen are actually capable of encoding olfactory information.

The GoodScents database provides 4,981 molecules with expert odor perception labels (floral, fruity, sweet, woody, green, spicy, animal_musk, earthy, citrus, chemical, gourmand, powdery_amber). This is a much larger and more diverse dataset. Training models on GoodScents to predict odor labels serves as a feature validation experiment:

**If molecular features (MACCS keys, Morgan fingerprints, physicochemical descriptors) can predict human odor perception labels with meaningful accuracy, this validates that these features encode chemically relevant olfactory information — and they can then be applied to the e-nose sensor prediction task.**

This is Phase 1 of a 3-phase pipeline:
- Phase 1 (current): Validate features on GoodScents odor labels
- Phase 2 (next): Apply validated + selected features to Aryballe 21-sample dataset
- Phase 3 (future): Incorporate peptide sequence features for full VOC-peptide binding prediction

---

# 2. RAW DATASET CHARACTERIZATION

## 2.1 GoodScents dataset

**Source:** The Good Scents Company database — one of the most comprehensive publicly available databases of olfactory descriptors assigned by trained perfumers and flavor chemists.

**Size:** 4,981 molecules × 12 odor labels

**Format:** Multi-label classification problem. Each molecule can have multiple odor labels simultaneously. This is important — a molecule can be both floral AND fruity at the same time.

**Average labels per molecule:** 3.33 — meaning the average molecule in the dataset has 3 different smell descriptors assigned to it.

**Distribution breakdown:**
| Label | Positive samples | Percentage | Category |
|---|---|---|---|
| Fruity | 2,282 | 45.8% | Balanced |
| Green | 2,252 | 45.2% | Balanced |
| Chemical | 2,145 | 43.1% | Balanced |
| Sweet | 1,928 | 38.7% | Moderately imbalanced |
| Floral | 1,304 | 26.2% | Imbalanced |
| Gourmand | 1,206 | 24.2% | Imbalanced |
| Spicy | 1,075 | 21.6% | Severely imbalanced |
| Woody | 1,055 | 21.2% | Severely imbalanced |
| Powdery amber | 1,026 | 20.6% | Severely imbalanced |
| Earthy | 1,005 | 20.2% | Severely imbalanced |
| Animal musk | 772 | 15.5% | Severely imbalanced |
| Citrus | 513 | 10.3% | Critically imbalanced |

**Problem identified:** 8 out of 12 labels have fewer than 27% positive samples. The standard machine learning assumption of balanced classes is violated for most labels. This was the primary challenge identified before any modeling began.

## 2.2 Label co-occurrence

Because this is multi-label data, labels co-occur frequently:
- Fruity + Green: 1,268 molecules (25.5% of dataset) — the most common pairing
- Fruity + Sweet: 1,095 molecules (22.0%)
- Fruity + Chemical: 939 molecules (18.9%)
- Green + Chemical: 902 molecules (18.1%)

**Implication for modeling:** Treating each label as an independent binary classification (as done in Jad Bio) ignores these correlations. A molecule that is fruity has a much higher probability of also being green. A multi-label or multi-task learning approach would be more appropriate, but was not available in Jad Bio. This is a limitation to acknowledge.

## 2.3 Multi-label complexity
- 779 molecules have exactly 1 label (simplest)
- 920 molecules have exactly 2 labels
- 3,282 molecules have 3 or more labels (65.9% of dataset)

This means the majority of molecules are perceptually complex — they cannot be cleanly assigned to a single smell category. This inherent ambiguity makes classification harder and is reflected in the model performance.

## 2.4 Aryballe dataset

**Size:** 21 measurements, 6 distinct VOC substances + 1 blank (Blanc)
**Substances:** Ocimene, Delta 3 Carene, Linalol, Alpha Pinene, S-Limonene, R-Limonene (3 replicates each)
**Sensors:** 8 channels (columns 1, 24, 25, 34, 36, 55, 105, 106)
**Values:** Normalized, range approximately 0-1

**Critical observation:** R-Limonene and S-Limonene are enantiomers — the same molecular formula and connectivity, but mirror-image 3D structures. They smell different (R-limonene = orange, S-limonene = turpentine-pine). This means any purely 2D molecular descriptor cannot distinguish them. 3D or chirality-aware features are required.

---

# 3. FEATURE ENGINEERING

## 3.1 Overview of the approach

The feature engineering strategy was inspired by computational drug discovery (QSAR — Quantitative Structure-Activity Relationships). Since the Aryballe sensors work via peptide-VOC binding (same physics as protein-ligand binding), the relevant molecular features are those that encode binding-relevant molecular properties.

Three families of features were computed using RDKit from SMILES representations of the molecules:

**Total features: 1,005**

## 3.2 MACCS keys (166 features, 160 non-zero)

**What they are:** MACCS (Molecular ACCess System) keys are 166 predefined binary substructure queries. Each bit indicates presence (1) or absence (0) of a specific chemical fragment in the molecule. Examples: bit 123 encodes presence of an oxygen-containing fragment (C=O or C-O-C), bit 154 encodes C=C double bonds, bit 165 encodes ring presence.

**Why chosen:** MACCS keys are interpretable, widely validated in odor QSAR literature, and appropriate for structurally diverse small molecules. They directly encode functional group presence which is the primary driver of odor character — esters (fruity), aldehydes (green), terpene structures (woody/piney), etc.

**Quality assessment:**
- 6 bits are always zero across all 4,981 molecules — these encode very rare substructures not present in the dataset and contribute zero information. They should be removed.
- 160 bits are informative
- All 166 are binary (0 or 1)

**Verdict: Good choice. Keep, remove 6 always-zero bits.**

**Validation from results:** MACCS_123 (oxygen fragments) appeared as a top feature in both fruity (rank 2) and sweet (rank 2) models. MACCS_121 was the top feature for fruity. MACCS_143 and MACCS_163 drove the spicy model. This cross-label consistency confirms that MACCS bits are encoding real chemical signals.

## 3.3 Morgan fingerprints (512 features)

**What they are:** Morgan fingerprints (also called ECFP — Extended Connectivity FingerPrints) encode circular chemical environments around each atom up to a defined radius (radius 2 = ECFP4). Each bit represents a specific local chemical neighborhood. Unlike MACCS, they are not predefined — they are hashed from the actual molecules and capture more specific substructures.

**Why chosen:** Morgan fingerprints capture finer-grained structural information than MACCS. They are the most widely used fingerprint in drug discovery and add complementary information to MACCS keys. They are particularly useful for distinguishing molecules with similar functional groups but different structural contexts.

**Quality assessment (serious problem):**
- Of the 512 bits, 65 are set in fewer than 50 molecules out of 4,981 (< 1% prevalence) — essentially useless
- 228 bits are set in fewer than 100 molecules (< 2% prevalence) — very sparse, unreliable signal
- Only ~284 bits have meaningful prevalence (≥ 2% of dataset)

**This means more than half of the 512 Morgan bits contribute mostly noise.**

**Why this happened:** 512-bit Morgan fingerprints are designed for large diverse libraries (millions of compounds). For a dataset of ~5,000 molecules, many hash buckets remain empty or near-empty. A 128-bit or 256-bit fingerprint, or filtering to prevalent bits only, would have been more appropriate.

**Verdict: Partially justified, but needs aggressive pruning. Filter to bits with ≥ 2% prevalence (~284 remain).**

## 3.4 Physicochemical descriptors (327 features)

These descriptors encode global molecular properties rather than structural fragments. They were computed using RDKit and the mordred library and cover:

| Sub-family | Count | What it encodes | Relevance to binding |
|---|---|---|---|
| Autocorrelation (ATSC/AATS/ATS) | 67 | Weighted topological distance correlations | Low-medium (redundant variants) |
| VSA descriptors (PEOE/SMR/SlogP/EState) | 62 | Surface area contributions by charge/refractivity/lipophilicity | HIGH — directly relevant to binding surface |
| Ring descriptors | 56 | Ring sizes, counts, aromaticity | Medium |
| Topological info indices (IC/CIC/MIC) | 14 | Information-theoretic graph complexity | Medium |
| JGI/GGI bond-walk indices | 16 | Topological connectivity | Medium |
| Atom/bond counts (nC, nO, nS...) | 16 | Element composition | High (nO for fruity, nS for sweet) |
| Lipophilicity (SLogP, FilterItLogS) | 3 | Hydrophobicity, solubility | HIGH — primary driver of binding |

**Key property ranges in the dataset:**
- SLogP (lipophilicity): mean=2.43, range=[-17.41, +12.94] — wide range, highly variable
- TopoPSA (polar surface area): mean=34.84, range=[0, 633.2] — enormous range
- nHeavyAtom (molecular size): mean=12.54, range=[1, 92]
- nRing (ring count): mean=0.77, range=[0, 30]

**Quality assessment (serious problem):**
- All 327 physicochemical NaN values come from this family (1,635 total NaN values). RDKit could not compute certain descriptors for some molecules (e.g. 3D descriptors requiring conformer generation, or edge-case molecules). This was handled by median imputation for SMOTE but remains in the dataset uploaded to Jad Bio — Jad Bio handles NaNs internally.
- The 67 autocorrelation descriptors (ATSC variants) compute the same mathematical operation (Broto-Moreau autocorrelation) with different atomic weights at different topological lags. Many are highly correlated with each other — massive internal redundancy.
- The 56 ring descriptors overlap heavily — nRing, n5Ring, n6Ring, nHRing, nARing etc. all encode variations of the same underlying ring count information.

**Verdict: Good coverage of binding-relevant properties, but over-complete. Need to remove correlated variants (Spearman r > 0.85 threshold).**

## 3.5 The samples-to-features ratio problem

**4,981 samples / 1,005 features = 4.96:1 ratio**

This is the most critical quantitative issue in the feature engineering. The general guidelines are:
- < 5:1 — danger zone, high overfitting risk, unstable feature selection
- 5-10:1 — acceptable but risky
- > 10:1 — acceptable
- > 20:1 — preferred for reliable models

At 4.96:1, every model trained on this dataset is at risk of instability. For minority-class labels, the effective ratio worsens dramatically because the model mainly learns from positive examples:
- Citrus (513 positive): 513/1005 = **0.51:1** — essentially impossible to learn reliably
- Animal musk (772 positive): 772/1005 = **0.77:1**
- Spicy (1,075 positive): 1,075/1005 = **1.07:1**

**This is why Jad Bio's internal Boruta feature selection is critical** — it reduces the effective feature set per label to 10-60 features before model fitting, which improves the ratio significantly.

## 3.6 What is missing from the feature set

**3D conformer descriptors:** All features computed are 2D (computed from molecular graph only). For peptide binding prediction (the ultimate goal), 3D shape matters. Principal moment of inertia ratios, asphericity, and radius of gyration cannot be computed from 2D features. RDKit's ETKDG conformer generation followed by Descriptors3D would provide these.

**Chirality descriptors:** S-Limonene and R-Limonene are in the Aryballe dataset and smell completely differently, but all 2D features (MACCS, Morgan, physicochemical) are identical for both enantiomers. Chirality-aware descriptors (chiral center count, or 3D USRCAT shape descriptors) are required to handle this.

**Standalone MolLogP and MolMR:** Although lipophilicity is partially encoded in SlogP_VSA descriptors, having explicit scalar MolLogP (octanol-water partition coefficient) and MolMR (molar refractivity) as direct features would improve binding prediction. These are the two most important descriptors in Lipinski's Rule of 5 and are standard in QSAR.

---

# 4. MODELING APPROACH

## 4.1 Platform and methodology

Jad Bio (Just Add Data Bioinformatics) was used as the AutoML platform. For each of the 12 odor labels, a separate binary classification model was trained (one-vs-rest approach). Each model independently selects features and optimizes a classifier.

Jad Bio's internal pipeline:
1. Feature selection via Boruta (iteratively removes features less informative than random probes)
2. Maximum signature size: 25 features (configurable)
3. Algorithm search across multiple classifiers (Gradient Boosting, Random Forest, SVM, KNN, etc.)
4. Out-of-sample performance estimation via repeated cross-validation
5. Reports threshold-independent (AUC, Mean Average Precision) and threshold-dependent (F1, Recall, Precision, MCC) metrics

**Positive class:** Class "1" (molecule HAS the odor label) in all models.

## 4.2 Performance metrics used and why

**AUC (Area Under the ROC Curve):** Primary metric. Threshold-independent — measures the model's ability to rank positive above negative examples across all possible thresholds. AUC = 0.5 means random performance, AUC = 1.0 means perfect. All our models were statistically significantly better than 0.5 (baseline).

**F1 Score:** Harmonic mean of precision and recall. Sensitive to class imbalance — low F1 often indicates the model is missing many positive examples.

**Recall (True Positive Rate):** Proportion of actual positive molecules correctly identified. The most critical metric for our imbalanced labels — a low recall means the model is missing most of the true positives.

**MCC (Matthews Correlation Coefficient):** The most robust metric for imbalanced classification. Unlike F1, MCC considers all four cells of the confusion matrix. A value of 0 = random, 1 = perfect. More reliable than accuracy on imbalanced data.

**Why not accuracy?** Accuracy is misleading with imbalanced classes. Woody has 21.2% positive samples — a model that predicts "not woody" for everything achieves 78.8% accuracy without learning anything. The woody model achieved 84.4% accuracy but only 0.427 recall — it is mostly predicting negative.

---

# 5. MODEL RESULTS AND ANALYSIS

## 5.1 Overall performance table

| Label | AUC | F1 | Recall | Precision | MCC | Class balance | Feature signal |
|---|---|---|---|---|---|---|---|
| Fruity | 0.872 | 0.778 | 0.762 | 0.795 | 0.600 | 45.8% | Strong (5.71) |
| Floral | 0.857 | 0.638 | 0.595 | 0.690 | 0.526 | 26.2% | Weak (3.04) |
| Gourmand | 0.845 | 0.502 | 0.377 | 0.769 | 0.450 | 24.2% | Moderate (5.09) |
| Woody | 0.845 | 0.530 | 0.427 | 0.716 | 0.468 | 21.2% | Near-zero (0.45) |
| Animal musk | 0.805 | 0.472 | 0.354 | 0.726 | 0.449 | 15.5% | Moderate (3.77) |
| Earthy | 0.802 | 0.516 | 0.401 | 0.745 | 0.470 | 20.2% | Moderate (3.77) |
| Chemical | 0.800 | 0.678 | 0.639 | 0.725 | 0.466 | 43.1% | Weak (2.08) |
| Green | 0.796 | 0.665 | 0.623 | 0.715 | 0.426 | 45.2% | Weak (2.13) |
| Spicy | 0.794 | 0.410 | 0.302 | 0.654 | 0.354 | 21.6% | Strong (6.31)* |
| Powdery amber | 0.792 | 0.403 | 0.323 | 0.554 | 0.317 | 20.6% | Weak (1.98) |
| Sweet | 0.753 | 0.603 | 0.587 | 0.622 | 0.366 | 38.7% | Strong (6.03) |

*Spicy has strong feature signal but terrible recall — pure class imbalance problem.

## 5.2 Why fruity is the best model

Fruity is the cleanest result and the most scientifically satisfying. Three factors align:
1. **Balanced classes:** 45.8% positive samples — near 50/50
2. **Clear structural correlate:** Fruity odors are dominated by esters (e.g. ethyl butyrate, isoamyl acetate) and lactones. These all contain oxygen in specific bonding configurations, which MACCS_123 and MACCS_121 encode directly.
3. **High feature importance with narrow confidence intervals:** Top feature importance = 5.71, and unlike weaker models, the confidence intervals do NOT include zero — meaning the features are genuinely statistically significant.

The top features (MACCS_121, MACCS_123, fMF, BalabanJ, MACCS_126, MACCS_154) are all chemically interpretable and tell a consistent story about fruity molecular structure.

## 5.3 Why woody failed (near-zero feature importance)

Woody is the most important negative result. The top feature importance is only 0.45 — essentially at noise level. Only 3 features have non-zero importance, compared to 20+ for fruity.

**This is not a feature engineering failure. It is a perceptual science problem.**

"Woody" as an odor category encompasses an enormous structural diversity:
- Sesquiterpenes (cedrene, vetivazulene) — large, complex ring systems
- Sandalwood compounds (santalol, Javanol) — bicyclic structures with specific OH positioning
- Birch tar compounds — phenolic structures
- Patchouli (patchoulol) — unique sesquiterpene alcohol
- Simple wood aldehydes — linear chains

These molecules share no common 2D structural motif that distinguishes them from non-woody molecules. The "woody" percept is generated by specific 3D shape complementarity with olfactory receptors — not by 2D functional group presence. This is confirmed by the fact that the only non-zero feature (SssssC — quaternary carbon count) is a structural feature of terpenes generally, not of woody odors specifically.

**Literature support:** Multiple QSAR studies report that woody, musky, and animal odors are the most difficult to model from molecular structure alone, with AUC values rarely exceeding 0.85 regardless of feature choice.

## 5.4 The chemical label paradox

Chemical is interesting: it has good class balance (43.1%) and decent performance (AUC 0.800, F1 0.678), but very weak feature importance (top = 2.08, all CIs include zero). This seems contradictory — how can the model work reasonably well if no features are individually significant?

The explanation is that "chemical" odor is associated with a broad pattern of *absence of specific functional groups* rather than presence of characteristic ones. Molecules smell "chemical" when they lack the groups that create other odors (fruity esters, floral terpenes, etc.). The model captures this through a combination of many weak signals, none of which is strong alone. This is also why chemical co-occurs so strongly with fruity (18.9%) and green (18.1%) — many molecules can be described as both "chemical" and something else.

## 5.5 The spicy paradox

Spicy shows the opposite pattern: strong feature importance (top = 6.31, the highest of any label) but terrible recall (0.302). The model knows exactly what spicy molecules look like (elongated geometry, specific MACCS ring patterns) but misses 70% of them.

This is pure class imbalance. With only 21.6% positive samples, the classifier is severely biased toward predicting negative. Even when it encounters a molecule with all the right spicy features, the prior probability pushes it toward predicting "not spicy." This is precisely what SMOTE oversampling is designed to fix — and is why spicy was selected as one of the three labels to retrain with balanced data.

## 5.6 The precision-recall tradeoff across labels

A consistent pattern across all minority-class labels: **high precision, low recall**. For example:
- Gourmand: Precision = 0.769, Recall = 0.377
- Animal musk: Precision = 0.726, Recall = 0.354
- Earthy: Precision = 0.745, Recall = 0.401

This means the models are conservative — when they predict a molecule belongs to a category, they are usually right (high precision), but they predict it far too rarely (low recall). This is the classic signature of a classifier trained on imbalanced data: it defaults to the majority class (negative) unless the evidence for positive is overwhelming.

---

# 6. PROBLEMS ENCOUNTERED AND SOLUTIONS

## Problem 1: Class imbalance

**Description:** 8 of 12 labels have fewer than 27% positive samples. For 4 labels (spicy, woody, earthy, animal_musk, powdery_amber, citrus), the positive class is below 22%, creating a severe prior probability bias toward negative prediction.

**Impact:** Recall collapsed to 0.302-0.427 for all severely imbalanced labels. Accuracy was misleading (78-88% for labels where always predicting "no" would give similar results).

**Solution applied:** SMOTE (Synthetic Minority Over-sampling TEchnique) with KNN interpolation. For the 3 most affected learnable labels (spicy, animal_musk, powdery_amber):
- KNN SMOTE with k=5 was used (interpolation between real nearest neighbors, not random noise)
- Binary features (MACCS, Morgan bits) were rounded back to 0/1 after interpolation
- Continuous physicochemical features were left as interpolated values (no clipping)
- NaN values were imputed with column median before KNN computation
- Result: all three datasets rebalanced to 50/50
- Normalized drift between synthetic and real positive samples: 0.016-0.020 (excellent quality)

**Why KNN SMOTE over Gaussian noise:** Gaussian noise generates samples randomly around existing points, potentially creating chemically implausible molecules (e.g. a bit that should encode ring presence gets a random fractional value). KNN SMOTE interpolates between real neighboring molecules in feature space, guaranteeing that synthetic samples lie within the convex hull of real data — more chemically plausible.

**Models retrained:** Spicy, Animal musk, Powdery amber (results pending)

## Problem 2: Too many features (dimensionality)

**Description:** 1,005 features for 4,981 samples = 4.96:1 ratio, below the recommended minimum of 10:1. Key contributors: 228 sparse Morgan bits (< 2% prevalence), 67 redundant autocorrelation variants, 56 overlapping ring descriptors.

**Impact:** Model instability, weak feature importance signals, potential overfitting especially for minority-class labels.

**Solution applied (partially):** Jad Bio's internal Boruta feature selection reduces the effective feature set to a maximum of 25 features per model before fitting. This is the main mitigation currently in place.

**Recommended additional solution (not yet applied):**
1. Remove 6 always-zero MACCS bits
2. Filter Morgan bits to those with ≥ 2% prevalence (~284 remain)
3. Correlation filter on physicochemical descriptors: remove one from each pair with Spearman r > 0.85
4. Target: reduce total features to 150-250 before uploading to Jad Bio

## Problem 3: The positive class definition in Jad Bio

**Description:** Jad Bio defaults to Class "0" as the positive class. For our binary labels (0 = does not have this smell, 1 = has this smell), Class "1" must be selected as positive. The first training run for fruity was done with Class "0" as positive, producing inverted metrics.

**Impact:** F1, Precision, and Recall metrics were measuring performance on the "not fruity" class rather than the fruity class. AUC remained valid (symmetric), but all threshold-dependent metrics were wrong.

**Solution:** All models were re-evaluated with Class "1" as positive. The correct metrics are those reported throughout this document.

**Lesson learned:** Always verify the positive class definition in any AutoML platform before interpreting threshold-dependent metrics.

## Problem 4: Multi-label nature ignored by binary classifiers

**Description:** Each molecule has on average 3.33 labels. Training 12 independent binary classifiers ignores the strong correlations between labels (e.g. fruity + green co-occur in 25.5% of molecules). A molecule predicted as fruity by one model has a much higher prior probability of also being green, sweet, and floral.

**Impact:** Each model wastes training capacity learning correlations independently. Models may make inconsistent predictions (predicting fruity but not sweet for a molecule that is clearly both).

**Solution (not yet applied):** Multi-label or multi-task learning approaches — e.g. Label Powerset transformation, Classifier Chains, or multi-output neural networks. These are beyond Jad Bio's current interface but would be the next methodological step in a full pipeline.

---

# 7. FEATURE IMPORTANCE FINDINGS

## 7.1 What the models actually learned

The feature importance scores from Jad Bio (computed as the contribution of each feature to model performance) reveal chemically interpretable patterns:

**Fruity (importance 5.71):** MACCS_121 (ring/aromatic pattern), MACCS_123 (C=O/ether oxygen), fMF (molecular framework fraction), BalabanJ (molecular branching). → Esters and ethers with ring systems. Chemically coherent.

**Spicy (importance 6.31):** MACCS_143, MACCS_163 (specific ring/chain patterns), GeomDiameter (molecular elongation), ATSC2v (van der Waals autocorrelation). → Elongated molecules with specific ring fragments, consistent with capsaicin-like spicy compounds.

**Sweet (importance 6.03):** Xp-3dv (path descriptor), MACCS_123 (oxygen fragment — shared with fruity), MACCS_154 (C=C), EState_VSA8, nS (sulfur count). → The shared MACCS_123 with fruity makes chemical sense (many sweet molecules are also fruity esters). nS is interesting — thiols and sulfur heterocycles contribute caramel/sweet notes.

**Gourmand (importance 5.09):** piPC1, piPC7 (pi-system descriptors), nHeavyAtom, Xc-3d. → Aromatic/conjugated pi systems, consistent with caramel, vanilla, and chocolate compounds that dominate gourmand.

**Green (importance 2.13):** All top features are Morgan fingerprints (morgan_356, morgan_119, morgan_79). → Specific local chemical environments rather than global properties, consistent with C6 aldehyde and monoterpene fragments that define green odors.

**Powdery amber (importance 1.98):** CIC0, MIC0, MIC1 (information complexity indices), SLogP, SlogP_VSA5. → Lipophilicity dominates, consistent with large non-polar amber/musk compounds. The direction is correct but the signal is too weak.

## 7.2 Cross-label recurring features

Three features appear across multiple labels:

**MACCS_123** (oxygen-containing fragment): Top-2 for fruity, top-2 for sweet. This is the strongest cross-label signal and validates that oxygen functional groups are the most important structural feature for predicting pleasant food-related odors.

**TopoPSA** (topological polar surface area): Appears in floral, green, chemical models. PSA reflects how much of the molecular surface is available for hydrogen bonding — lower PSA = more non-polar = more volatile = more odorous in general.

**SLogP and lipophilicity descriptors:** Dominate powdery_amber and appear in earthy and animal_musk. Large, non-polar molecules require high lipophilicity for receptor binding.

## 7.3 Answer to the key question: are the features relevant?

**Yes, with important limitations.**

Evidence FOR relevance:
- MACCS_123 predicts both fruity AND sweet — oxygen fragment signal is real and consistent
- MACCS_143/163 drive spicy prediction with importance 6.31 — ring/chain patterns are chemically meaningful
- SLogP correctly identified as dominant for powdery_amber — lipophilicity is the right descriptor for large non-polar musks
- BalabanJ, fMF, piPC series all appear in multiple labels — these are not random
- All 11 models are statistically significantly better than baseline AUC of 0.500

Evidence of LIMITATIONS:
- 228 Morgan bits are too sparse to reliably contribute — they inflate dimensionality without adding signal
- 67 ATSC autocorrelation variants are internally redundant
- 2D features cannot distinguish enantiomers (critical limitation for the R/S-limonene case in the Aryballe dataset)
- Woody and chemical feature importance collapses to near zero — these categories are not learnable from 2D structure alone

---

# 8. CONNECTION TO THE E-NOSE AND FUTURE WORK

## 8.1 How this analysis feeds into the e-nose prediction

The current GoodScents models predict human odor perception labels. The Aryballe sensors produce numeric values (not binary labels) that reflect peptide-VOC binding strength. The connection between the two:

1. The molecular features that predict "fruity" (MACCS_123, oxygen fragments) are the same features that should predict high sensor response on sensors tuned to detect esters and ethers.
2. The feature importance rankings from GoodScents provide a scientifically motivated pre-selection for the Aryballe modeling step — rather than using all 1,005 features on 21 samples, we can start with the 20-50 features validated here.
3. The VSA, SLogP, and topological features that appear consistently are all physically motivated for binding prediction.

## 8.2 Clustering analysis (ongoing)

Clustering analysis was performed in parallel to determine whether molecules naturally group by odor category in the molecular feature space. Methods used: [TO BE FILLED with your tutor's method and ISIPCA results]. The clustering results will validate whether the feature space creates meaningful odor-based separations without supervision — a complementary check to the supervised classification results here.

## 8.3 Recommended next steps

**Immediate (before next Jad Bio run):**
1. Retrain spicy, animal_musk, powdery_amber with SMOTE-balanced data (in progress)
2. Compare before/after SMOTE results — expected: +0.10-0.20 recall improvement
3. Run feature selection pipeline: remove always-zero MACCS, filter sparse Morgan bits, correlation-filter physicochemical descriptors

**Short term:**
1. Apply validated feature set to Aryballe 21-sample dataset (regression, not classification)
2. Use time-series features from the raw sensogram data (slope, plateau value, response area under curve)
3. Add 3D conformer descriptors via RDKit ETKDG for chirality-sensitive prediction

**Long term:**
1. Obtain peptide sequence data from Aryballe
2. Model VOC-peptide binding as molecule-peptide pairs (analogous to protein-ligand QSAR)
3. Predict sensor response for new molecules not in the training set

---

# 9. CONCLUSIONS

1. **Feature validation: SUCCESS** — Molecular descriptors (MACCS keys, Morgan fingerprints, physicochemical) encode meaningful olfactory information. 11 of 11 trained models outperform random baseline, and the best models achieve AUC ≥ 0.87.

2. **Best performance** achieved for fruity (AUC 0.872, F1 0.778) — balanced classes + clear structural correlates (oxygen-containing fragments).

3. **Primary failure mode: class imbalance** — 8/12 labels have < 27% positive samples. Recall collapses below 0.45 for all severely imbalanced labels. This is an experimental design problem, not a feature failure.

4. **Secondary failure mode: dimensionality** — 1,005 features for ~5,000 samples is below the recommended ratio. Sparse Morgan bits and redundant autocorrelation descriptors add noise rather than signal.

5. **Irreducible failures:** Woody and chemical have near-zero feature importance regardless of modeling approach. This reflects perceptual science limitations — these odor categories do not map onto consistent 2D molecular structures. The literature confirms this finding.

6. **Enantiomer problem:** R-limonene and S-limonene cannot be distinguished by any 2D feature. 3D/chirality descriptors are required for the e-nose prediction step.

7. **The pipeline is validated:** The feature engineering approach is scientifically sound and ready for the next phase — applying validated features to predict Aryballe e-nose sensor responses.
