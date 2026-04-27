# Internship Project: Predicting E-Nose Sensor Responses from Molecular Features

[cite_start]This project sits at the intersection of **digital olfaction**, **cheminformatics**, and **machine learning**[cite: 5]. [cite_start]The goal is to build predictive models that take the molecular structure of a Volatile Organic Compound (VOC) and predict the response intensity of an Aryballe electronic nose's sensors[cite: 6].

---

## 1. Project Overview
[cite_start]Unlike standard scent classification ("What is this smell?"), this is a **regression problem**[cite: 7]. 

* [cite_start]**Input:** Molecular features of a substance[cite: 8].
* [cite_start]**Output:** Predicted sensor response values (ΔR%)[cite: 8, 26].

[cite_start]The model predicts sensor values directly from molecular chemistry[cite: 9].

---

## 2. Technology & Binding Mechanism
### 2.1 The Aryballe NeOse Pro
[cite_start]The device uses **Surface Plasmon Resonance imaging (SPRi)**[cite: 12]. [cite_start]Its silicon chip is grafted with an array of **peptide biosensors** (4–7 amino acid residues) that bind to VOC molecules in the gas phase[cite: 13].
* [cite_start]**Measurement:** Binding events change the refractive index, captured by an optical camera as changes in reflectivity (**ΔR%**)[cite: 14, 15].
* [cite_start]**Sensors:** The dataset tracks 8 specific sensor channels (1, 24, 25, 34, 36, 55, 105, 106)[cite: 25, 26].

### 2.2 Feature Selection Logic
[cite_start]Peptide-VOC binding is governed by physical forces identical to drug discovery (Hydrogen bonds, Van der Waals, etc.)[cite: 18, 19]. Relevant descriptors include:
* [cite_start]**Hydrophobicity:** MolLogP, TPSA[cite: 20].
* [cite_start]**Size/Polarizability:** MolMR, LabuteASA[cite: 21].
* [cite_start]**H-bonding:** NumHDonors, NumHAcceptors[cite: 22].
* [cite_start]**Shape/Flexibility:** NumRotatableBonds, BertzCT[cite: 23].

---

## 3. Dataset Summary
[cite_start]The current test dataset includes 6 unique molecules used to validate the pipeline[cite: 32]:
* [cite_start]**Molecules:** Ocimene, Delta-3-Carene, Linalool, Alpha-Pinene, S-Limonene, R-Limonene[cite: 31].
* **Files:**
    * [cite_start]`7Q27_normalized_signatures.csv`: Pre-processed sensor responses (21 rows)[cite: 29].
    * [cite_start]`7Q27_sensograms.csv`: Raw time-series data (62,390 rows)[cite: 29].

---

## 4. Feature Engineering Pipeline
| Stage | Task | Status |
| :--- | :--- | :--- |
| 1. Raw Data | Obtain SMILES from PubChem for each substance | [cite_start]In progress [cite: 36] |
| 2. Molecular Descriptors | Compute RDKit/Mordred physicochemical descriptors | [cite_start]Pipeline ready [cite: 36] |
| 3. Fingerprints + Spectra | Compute MACCS/Morgan fingerprints & mass spectral features | [cite_start]Pipeline ready [cite: 36] |
| 4. Dataset Assembly | Merge features with e-nose measurements | [cite_start]Pipeline ready [cite: 36] |
| 5. Preprocessing | Drop zero-variance, KNN impute, scale | [cite_start]Pipeline ready [cite: 36] |
| 6. Jad Bio Modeling | Train regression models (one per sensor or multi-output) | [cite_start]Waiting for full dataset [cite: 36] |

---

## 5. Modeling Setup
[cite_start]Modeling is performed using **Jad Bio**, an AutoML platform[cite: 57, 58].
* [cite_start]**Strategy:** One model per sensor (8 total) or multi-output regression[cite: 60].
* [cite_start]**Preprocessing:** 1.  Drop near-zero variance columns[cite: 62].
    2.  [cite_start]KNN Imputation (n_neighbors=3)[cite: 63].
    3.  [cite_start]MinMax Scaling [0, 1][cite: 64].
    4.  [cite_start]Feature selection (Boruta or LASSO) for larger datasets[cite: 56, 65].

---

## 6. Required Tools
| Tool | Purpose | Install |
| :--- | :--- | :--- |
| **RDKit** | Molecular descriptors and fingerprints | [cite_start]`pip install rdkit` [cite: 67] |
| **Mordred** | 1800+ automated descriptors | [cite_start]`pip install mordred` [cite: 67] |
| **pandas** | Dataset assembly | [cite_start]`pip install pandas` [cite: 67] |
| **scikit-learn** | Imputation and scaling | [cite_start]`pip install scikit-learn` [cite: 67] |
| **PubChem API** | Fetching SMILES strings | [cite_start]HTTP requests [cite: 67] |

---

## 7. Next Steps
1.  [cite_start]Finalize SMILES for the initial 6 substances[cite: 76].
2.  [cite_start]Run the RDKit + MACCS pipeline as a test[cite: 77].
3.  [cite_start]Download EI mass spectra from NIST WebBook[cite: 78].
4.  [cite_start]Assemble enriched test CSV and wait for the full multi-substance dataset[cite: 79, 80].
