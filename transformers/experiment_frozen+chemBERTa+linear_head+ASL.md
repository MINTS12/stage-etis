# Experiment 01 — Frozen ChemBERTa + Linear Head + ASL

**Goal:** Classify 12 odor metacategories and beat the BR+RF baseline (macro PR AUC = 0.624)  
**Dataset:** GoodScents + Leffingwell · 4,981 molecules  
**Script:** `First_experiment.py`

---

## Motivation

Traditional molecular fingerprints (MACCS, Morgan) encode structure through hand-crafted bit patterns. A transformer pretrained on 77M SMILES strings (ChemBERTa) may learn richer, context-aware representations — capturing long-range atom interactions that fixed fingerprints miss.

The hypothesis: **better molecular representation → better odor label prediction**.

The backbone is kept **frozen** to avoid overfitting a 77M-parameter model on only ~5K molecules. Only a lightweight classification head is trained on top of the fixed embeddings.

---

## Method

### 1. SMILES → embedding

Each SMILES string is tokenized and passed through the frozen ChemBERTa backbone (`seyonec/ChemBERTa-zinc-base-v1`, RoBERTa architecture). The `[CLS]` token at position 0 of the last hidden state is taken as the molecule's fixed representation:

```
h = ChemBERTa(SMILES)[CLS]   ∈ ℝ^768
```

Embeddings are extracted once upfront and cached in memory — since the backbone is frozen, the same SMILES always produces the same vector.

### 2. Linear classification head

A single linear layer maps the 768-dim embedding to 12 label logits. Sigmoid is applied independently per label (multi-label setting, not softmax):

```
ŷ = σ(W · h + b),   W ∈ ℝ^{12×768},   ŷ ∈ (0,1)^12
```

### 3. Asymmetric Loss (ASL)

Standard Binary Cross-Entropy is dominated by easy negatives in imbalanced multi-label settings. ASL decouples positives and negatives with separate focusing exponents, and applies a probability margin `m` to hard-discard very easy negatives from the gradient:

```
L_pos = − y  · (1 − p)^{γ+} · log(p)
L_neg = − (1−y) · p_m^{γ−} · log(1 − p_m)

where p_m = max(p − m, 0)
```

**Hyperparameters used:** γ⁺ = 0, γ⁻ = 4, m = 0.05

This directly benefits minority labels (citrus, animal_musk, powdery_amber) where easy negatives would otherwise dominate the gradient.

### 4. Per-label threshold calibration

Rather than a fixed 0.5 decision threshold, an optimal threshold is searched per label on the validation set to maximise per-label F1:

```
t*_i = argmax_{t ∈ [0.1, 0.9]} F1(y_i, ŷ_i ≥ t)
```

Applied at test time. This is the dominant lever on macro F1 but does not affect PR AUC or ROC AUC.

### Training configuration

| Parameter | Value |
|---|---|
| Model | `seyonec/ChemBERTa-zinc-base-v1` |
| Backbone | Frozen (no gradient) |
| Head | Linear (768 → 12) |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Epochs | 300 |
| Batch size | 64 |
| LR schedule | CosineAnnealingLR |
| Split | 60 / 20 / 20 (multilabel stratified) |

---

## Results

| Metric | This experiment | BR+RF baseline |
|---|---|---|
| **Macro PR AUC** | **0.548** | **0.624** |
| Macro ROC AUC | 0.761 | 0.820 |
| Macro F1 | 0.545 | 0.567 |

### Per-label PR AUC

| Label | PR AUC |
|---|---|
| fruity | 0.748 |
| chemical | 0.679 |
| green | 0.676 |
| sweet | 0.607 |
| floral | 0.595 |
| gourmand | 0.578 |
| earthy | 0.537 |
| woody | 0.517 |
| spicy | 0.462 |
| powdery_amber | 0.412 |
| animal_musk | 0.394 |
| citrus | 0.377 |

---

## Key finding

Frozen ChemBERTa embeddings **do not outperform** MACCS + Morgan + Mordred with BR+RF (0.548 vs 0.624 macro PR AUC).

The model converges early (~epoch 35, val PR AUC ≈ 0.529) and plateaus for the remaining 265 epochs — indicating the linear head has exhausted the information available in the fixed embedding space. Weak labels (citrus, animal_musk, spicy, powdery_amber) are the primary drag on macro performance.

This result motivates **Step 2 (MLP head)** and **Step 3 (fine-tuning)**: the pretrained representations carry structural chemistry knowledge but need task-specific adaptation to encode odor-relevant features that fingerprints already capture implicitly.
