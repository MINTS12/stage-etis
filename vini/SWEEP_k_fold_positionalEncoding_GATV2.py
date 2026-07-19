# Generated from: SWEEP_positionalEncoding_GATV2_HMCNF.ipynb
# Converted at: 2026-07-06T14:15:27.121Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# --- Reproducibility: fix all sources of randomness ---------------------------
import os
import random
import pickle
import numpy as np
import torch

SEED = 42

def set_seed(seed=SEED):
    """Seed every RNG that affects this pipeline's stochastic behaviour."""
    random.seed(seed)
    np.random.seed(seed)          # also fixes skmultilearn's IterativeStratification,
                                   # whose internal tie-breaking falls back to the
                                   # global numpy RNG regardless of random_state (see
                                   # https://github.com/scikit-multilearn/scikit-multilearn/issues/144)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
# -------------------------------------------------------------------------------

# --- Runtime tracking -----------------------------------------------------------
from datetime import datetime
SCRIPT_START_TIME = datetime.now()
print(f"Script started at: {SCRIPT_START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
# ---------------------------------------------------------------------------------

# --- DataLoader workers: OS-aware ----------------------------------------------
# Windows uses 'spawn' to start DataLoader worker processes, which re-imports this
# whole script as a module. Since this file has top-level executable code (not just
# defs) outside an `if __name__ == "__main__":` guard, that re-import recurses and
# crashes (RuntimeError from multiprocessing.spawn). Linux (this script's target
# server) uses 'fork' instead, which doesn't have this problem, so num_workers=4
# there is safe and fine to keep for full parallel data loading.
NUM_WORKERS = 0 if os.name == "nt" else 4
PERSISTENT_WORKERS = NUM_WORKERS > 0   # persistent_workers requires num_workers > 0
# -------------------------------------------------------------------------------

# # **Installing Dependencies**


# !pip install torch_geometric
# !pip install rdkit
# !pip install scikit-learn matplotlib

# # Parameters


metrics_path = "gatv2_hmcnf_metrics_new_features.csv"

batch_size = 128
num_heads = 8
lambda_hier = 0
learning_rate = 0.001
t_0_cosine = 50
epochs = 1000

# Format:
# [conv1_out, conv2_out, conv3_out, conv4_out,
#  global_mlp1_out, global_mlp2_out,
#  local1_transition_out, local2_transition_out,
#  dropout]

dims_baseline = [15,  20,  27,  36,   96,  63,  48,  63,  0.47]
dims_A        = [32,  48,  64,  96,  128, 138,  64, 138,  0.47]
dims_B        = [64,  96, 128, 160,  256, 256, 128, 256,  0.50]
dims_C        = [64,  96, 128, 192,  256, 256, 128, 256,  0.55]

dims = dims_C

# # Using Hierarchical Multi-Label Classification method with the 138 labels and the 12 groups simultaniously


# ## **Dataset Preprocessing**


# ### Importing the dataset


import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/fperone/Projet_IA-Generative-pour-la-decouverte-moleculaire/refs/heads/main/Multi-Labelled_Smiles_Odors_dataset.csv')

df.head()

df_new_dataset = pd.read_csv('https://raw.githubusercontent.com/Baiaopires/Internship_project/refs/heads/main/our_data/7Q27.normalized.signatures_with_smiles.csv')
df_new_dataset = df_new_dataset.head(-3)['SMILES']
new_molecules = df_new_dataset.unique()
new_molecules

df.shape

# Correlation


# Removing the molecules in `new_molecules` from the dataframe for testing later


df_filtered = df[~df['nonStereoSMILES'].isin(new_molecules)]

print(f"Original DataFrame shape: {df.shape}")
print(f"New molecules to remove count: {len(new_molecules)}")
print(f"Filtered DataFrame shape: {df_filtered.shape}")
df = df_filtered.reset_index(drop=True).copy()

# ### Checking for NaNs in the dataset


df.isna().sum()

# Substituting NaNs for Zeros


df.fillna(0, inplace=True)
df.isna().sum()

# ### Defining the 12 groups of smells and grouping the labels accordingly


# Not differentiating the macro and micro categories


META_CATEGORIES = {
    "floral": ["floral", "rose", "jasmin", "lily", "muguet", "violet", "hyacinth",
               "geranium", "lavender", "orangeflower", "chamomile", "hawthorn"],
    "fruity": ["fruity", "apple", "apricot", "banana", "berry", "cherry", "grape",
               "grapefruit", "lemon", "melon", "orange", "peach", "pear", "pineapple",
               "plum", "raspberry", "strawberry", "tropical", "black currant", "fruit skin"],
    "sweet": ["sweet", "vanilla", "caramellic", "honey", "chocolate", "cocoa",
              "coconut", "creamy", "buttery", "milky", "dairy"],
    "woody": ["woody", "cedar", "sandalwood", "pine", "vetiver", "terpenic",
              "balsamic", "cortex"],
    "green": ["green", "grassy", "herbal", "leafy", "hay", "tea", "fresh",
              "cucumber", "vegetable", "weedy"],
    "spicy": ["spicy", "cinnamon", "clove", "warm", "pungent", "sharp",
              "cooling", "mint", "camphoreous"],
    "animal_musk": ["animal", "musk", "leathery", "fishy", "sweaty", "meaty",
                    "beefy", "musty"],
    "earthy": ["earthy", "mushroom", "nutty", "hazelnut", "roasted", "coffee",
               "tobacco", "smoky", "popcorn"],
    "citrus": ["citrus", "bergamot", "ozone", "clean", "soapy"],
    "chemical": ["solvent", "ethereal", "metallic", "medicinal", "phenolic",
                 "sulfurous", "gassy", "burnt", "oily"],
    "gourmand": ["almond", "malty", "rummy", "brandy", "cognac", "winey",
                 "cooked", "potato", "savory", "celery", "tomato", "radish",
                 "onion", "garlic", "cabbage", "cheesy"],
    "powdery_amber": ["amber", "powdery", "anisic", "coumarinic", "orris",
                      "waxy", "aldehydic", "ketonic", "lactonic"],
}

# Differentiating the macro and micro categories


META_CATEGORIES = {
    "macro_floral": ["floral", "rose", "jasmin", "lily", "muguet", "violet", "hyacinth",
                     "geranium", "lavender", "orangeflower", "chamomile", "hawthorn"],
    "macro_fruity": ["fruity", "apple", "apricot", "banana", "berry", "cherry", "grape",
                     "grapefruit", "lemon", "melon", "orange", "peach", "pear", "pineapple",
                     "plum", "raspberry", "strawberry", "tropical", "black currant", "fruit skin"],
    "macro_sweet": ["sweet", "vanilla", "caramellic", "honey", "chocolate", "cocoa",
                    "coconut", "creamy", "buttery", "milky", "dairy"],
    "macro_woody": ["woody", "cedar", "sandalwood", "pine", "vetiver", "terpenic",
                    "balsamic", "cortex"],
    "macro_green": ["green", "grassy", "herbal", "leafy", "hay", "tea", "fresh",
                    "cucumber", "vegetable", "weedy"],
    "macro_spicy": ["spicy", "cinnamon", "clove", "warm", "pungent", "sharp",
                    "cooling", "mint", "camphoreous"],
    "macro_animal_musk": ["animal", "musk", "leathery", "fishy", "sweaty", "meaty",
                          "beefy", "musty"],
    "macro_earthy": ["earthy", "mushroom", "nutty", "hazelnut", "roasted", "coffee",
                     "tobacco", "smoky", "popcorn"],
    "macro_citrus": ["citrus", "bergamot", "ozone", "clean", "soapy"],
    "macro_chemical": ["solvent", "ethereal", "metallic", "medicinal", "phenolic",
                       "sulfurous", "gassy", "burnt", "oily"],
    "macro_gourmand": ["almond", "malty", "rummy", "brandy", "cognac", "winey",
                       "cooked", "potato", "savory", "celery", "tomato", "radish",
                       "onion", "garlic", "cabbage", "cheesy"],
    "macro_powdery_amber": ["amber", "powdery", "anisic", "coumarinic", "orris",
                             "waxy", "aldehydic", "ketonic", "lactonic"],
}

# ### Parent-Child index Mapping


# Considering all child-parent pairs according to the META_CATEGORIES list, even if it doesn't make much sense with our dataset. Some labels that are the children of some parents don't have a big correlation, for example, $P(floral|rose)=0.78$, which does not make much sense, since all rose smells are floral


label_columns = list(df.columns[2:])

child_parent_pairs = []

for parent_idx, (group_name, child_names) in enumerate(META_CATEGORIES.items()):
    for child_name in child_names:
        if child_name in label_columns:
            child_col_idx = label_columns.index(child_name)
            child_parent_pairs.append((child_col_idx, parent_idx))

# Considering only the child-parent pairs that make sense, in which $(parent|child) ≥ 0.6$ (a reasonable threshold for "this child reliably implies its parent"), if that condition is not satisfied, the child-parent pair is not added to the final list of child-parent pairs. The labels that are not in that final list will not be penalised in the loss function


# CONDITIONAL_PROB_THRESHOLD = 0.6

# import pandas as pd
# import numpy as np

# df_labels = df.iloc[:, 2:]
# label_columns = list(df.columns[2:])
# validated_pairs = []

# for parent_idx, (group_name, child_names) in enumerate(META_CATEGORIES.items()):
#     for child_name in child_names:
#         if child_name == group_name:
#             continue
#         if child_name not in label_columns:
#             continue
#         child_col = df_labels[child_name]
#         if child_col.sum() == 0:
#             continue
#         if group_name not in label_columns:
#             continue
#         p_parent_given_child = df_labels.loc[child_col==1, group_name].mean()
#         if p_parent_given_child >= CONDITIONAL_PROB_THRESHOLD:
#             child_col_idx = label_columns.index(child_name)
#             validated_pairs.append((child_col_idx, parent_idx))

# print(f"Original pairs: {len(child_parent_pairs)}")
# print(f"Validated pairs (P >= {CONDITIONAL_PROB_THRESHOLD}): {len(validated_pairs)}")
# child_parent_pairs = validated_pairs

child_parent_pairs[:5]

# ### Converting the SMILES representation to a graph one


# #### Converting the SMILES to graphs with `from_smiles`


import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix

def compute_rwpe(edge_index, num_nodes, k=8):
    # Build row-normalized adjacency (random walk operator)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).toarray()
    deg = adj.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1  # avoid division by zero
    RW = adj / deg  # RW = A D^{-1}
    
    rwpe = []
    RW_power = RW.copy()
    for _ in range(k):
        rwpe.append(np.diag(RW_power))  # self-return probabilities
        RW_power = RW_power @ RW
    
    return torch.tensor(np.stack(rwpe, axis=1), dtype=torch.float)  # [num_nodes, k]

import torch
from rdkit import Chem
from rdkit.Chem import rdPartialCharges, MolFromSmarts
from rdkit.Chem.rdchem import HybridizationType, ChiralType, BondStereo, BondType
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# ---------------------------------------------------------------------------
# Pauling electronegativity lookup — used for bond polarity (edge feature 7)
# Defaults to C (2.55) for atoms not in the table (rare in fragrance molecules)
# ---------------------------------------------------------------------------
_EN = {
    'H': 2.20, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
    'Se': 2.55, 'As': 2.18, 'Te': 2.10,
}

# Pre-compile SMARTS patterns once — reused for every molecule
_HBD_SMARTS = MolFromSmarts('[#7,#8;!H0]')   # H-bond donor:    N or O carrying H
_HBA_SMARTS = MolFromSmarts('[#7,#8]')        # H-bond acceptor: any N or O


def smiles_to_graph(smiles: str, y_tensor: torch.Tensor) -> Data | None:
    """
    Convert a SMILES string into a PyTorch Geometric Data object with:
      - data.x          : node features  [N x 22]  (see table below)
      - data.edge_index : COO adjacency  [2 x 2E]  (bidirectional)
      - data.edge_attr  : edge features  [2E x 8]  (see table below)
      - data.y          : label tensor   [138]
      - data.smiles     : original SMILES string

    Returns None if RDKit cannot parse the SMILES.

    Node feature index map (22 dims)
    ---------------------------------
     0      Atomic number
     1–4    Chirality         one-hot [unspecified, CW, CCW, other]
     5      Degree            (number of explicit bonds)
     6      Formal charge
     7      Implicit H count
     8      Radical electrons
     9–12   Hybridisation     one-hot [SP, SP2, SP3, other]
     13     Is aromatic?      binary
     14–18  Ring size         binary flags [in 3-, 4-, 5-, 6-, 7-membered ring]
     19     Gasteiger partial charge  (continuous, NaN → 0.0)
     20     H-bond donor      binary
     21     H-bond acceptor   binary

    Edge feature index map (8 dims)
    ---------------------------------
     0–3    Bond type         one-hot [single, double, triple, aromatic]
     4      Is in ring?       binary
     5      Is stereo (E/Z)?  binary
     6      Is conjugated?    binary
     7      Bond polarity     |Δ electronegativity| (continuous, Pauling scale)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # --- molecule-level precomputation ---
    rdPartialCharges.ComputeGasteigerCharges(mol)

    donors    = {idx for match in mol.GetSubstructMatches(_HBD_SMARTS) for idx in match}
    acceptors = {idx for match in mol.GetSubstructMatches(_HBA_SMARTS) for idx in match}

    ring_info = mol.GetRingInfo()
    atom_ring_sizes: dict[int, set[int]] = {}
    for ring in ring_info.AtomRings():
        for idx in ring:
            atom_ring_sizes.setdefault(idx, set()).add(len(ring))

    # --- node features [N x 22] ---
    node_features = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        h   = atom.GetHybridization()
        c   = atom.GetChiralTag()
        rs  = atom_ring_sizes.get(idx, set())

        charge = atom.GetDoubleProp('_GasteigerCharge')
        if charge != charge:    # NaN guard (NaN != NaN is always True)
            charge = 0.0

        node_features.append([
            # 0       atomic number
            atom.GetAtomicNum(),
            # 1–4     chirality one-hot
            int(c == ChiralType.CHI_UNSPECIFIED),
            int(c == ChiralType.CHI_TETRAHEDRAL_CW),
            int(c == ChiralType.CHI_TETRAHEDRAL_CCW),
            int(c == ChiralType.CHI_OTHER),
            # 5       degree
            atom.GetDegree(),
            # 6       formal charge
            atom.GetFormalCharge(),
            # 7       implicit H count
            atom.GetTotalNumHs(),
            # 8       radical electrons
            atom.GetNumRadicalElectrons(),
            # 9–12    hybridisation one-hot
            int(h == HybridizationType.SP),
            int(h == HybridizationType.SP2),
            int(h == HybridizationType.SP3),
            int(h not in (HybridizationType.SP,
                          HybridizationType.SP2,
                          HybridizationType.SP3)),
            # 13      is aromatic
            int(atom.GetIsAromatic()),
            # 14–18   ring size membership
            int(3 in rs),
            int(4 in rs),
            int(5 in rs),
            int(6 in rs),
            int(7 in rs),
            # 19      Gasteiger partial charge
            charge,
            # 20      H-bond donor
            int(idx in donors),
            # 21      H-bond acceptor
            int(idx in acceptors),
        ])

    # --- edge features [2E x 8], both directions per bond ---
    src, dst = [], []
    edge_features = []

    for bond in mol.GetBonds():
        i  = bond.GetBeginAtomIdx()
        j  = bond.GetEndAtomIdx()
        bt = bond.GetBondType()

        en_i     = _EN.get(mol.GetAtomWithIdx(i).GetSymbol(), 2.55)
        en_j     = _EN.get(mol.GetAtomWithIdx(j).GetSymbol(), 2.55)
        en_delta = abs(en_i - en_j)

        feat = [
            # 0–3   bond type one-hot
            int(bt == BondType.SINGLE),
            int(bt == BondType.DOUBLE),
            int(bt == BondType.TRIPLE),
            int(bt == BondType.AROMATIC),
            # 4     is in ring
            int(bond.IsInRing()),
            # 5     is stereo (E/Z)
            int(bond.GetStereo() != BondStereo.STEREONONE),
            # 6     is conjugated
            int(bond.GetIsConjugated()),
            # 7     bond polarity
            en_delta,
        ]

        src += [i, j]
        dst += [j, i]
        edge_features += [feat, feat]   # same features for both directions

    x          = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    rwpe = compute_rwpe(edge_index, x.shape[0], k=8)
    x = torch.cat([x, rwpe], dim=-1)  # 22 + 8 = 30 node features
    
    edge_attr  = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=y_tensor, smiles=smiles)

df_graph = []
labels   = []
# G        = []
skipped  = []

for i in range(len(df)):
    smiles  = df['nonStereoSMILES'][i]
    y       = torch.tensor(df.iloc[i, 2:].to_numpy(dtype=float), dtype=torch.float)
    data    = smiles_to_graph(smiles, y)

    if data is None:                        # RDKit failed to parse SMILES
        skipped.append(i)
        continue

    df_graph.append(data)
    # G.append(to_networkx(data, to_undirected=True))
    labels.append(df.iloc[i, 2:])

if skipped:
    print(f"Warning: {len(skipped)} molecules skipped (unparseable SMILES): {skipped}")

print(f"Built {len(df_graph)} graphs")
print(f"Node feature dim : {df_graph[0].x.shape[1]}")   # should be 22
print(f"Edge feature dim : {df_graph[0].edge_attr.shape[1]}")  # should be 8

# ### Creating the train and test loaders


# #### Second-order iterative stratification converting the SMILES to graphs with `from_smiles`


# !pip install scikit-multilearn

import numpy as np
import pandas as pd
from skmultilearn.model_selection import IterativeStratification
from torch_geometric.loader import DataLoader
import torch
from torch_geometric.utils import from_smiles
from torch_geometric.utils import to_networkx

def create_stratified_splits(df, label_columns, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=SEED):
    np.random.seed(seed)   # workaround: skmultilearn's tie-breaking falls back to the global RNG
    X = np.arange(len(df)).reshape(-1, 1)
    y = label_columns

    holdout_ratio = val_ratio + test_ratio

    stratifier_1 = IterativeStratification(
        n_splits=2,
        order=2,
        sample_distribution_per_fold=[holdout_ratio, train_ratio],
    )

    train_idx, holdout_idx = next(stratifier_1.split(X, y))

    print(f"   -> Step 1 Complete: {len(train_idx)} Train samples, {len(holdout_idx)} Holdout samples.")

    relative_test_ratio = test_ratio / holdout_ratio
    relative_val_ratio = 1.0 - relative_test_ratio

    np.random.seed(seed)   # reseed before the second split, same reason as above
    stratifier_2 = IterativeStratification(
        n_splits=2,
        order=2,
        sample_distribution_per_fold=[relative_test_ratio, relative_val_ratio],
    )

    X_holdout = X[holdout_idx]
    y_holdout = y[holdout_idx]

    val_idx_relative, test_idx_relative = next(stratifier_2.split(X_holdout, y_holdout))

    val_idx = holdout_idx[val_idx_relative]
    test_idx = holdout_idx[test_idx_relative]

    print(f"   -> Step 2 Complete: {len(val_idx)} Val samples, {len(test_idx)} Test samples.")

    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()
    df_test = df.iloc[test_idx].copy()

    return df_train, df_val, df_test, train_idx, val_idx, test_idx


SPLIT_CACHE = "fixed_split_indices.pkl"

if os.path.exists(SPLIT_CACHE):
    with open(SPLIT_CACHE, "rb") as f:
        train_idx, val_idx, test_idx = pickle.load(f)
    train_data = df.iloc[train_idx].copy()
    val_data   = df.iloc[val_idx].copy()
    test_data  = df.iloc[test_idx].copy()
    print(f"Loaded cached split from '{SPLIT_CACHE}' "
          f"({len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test).")
else:
    train_data, val_data, test_data, train_idx, val_idx, test_idx = create_stratified_splits(
        df=df,
        label_columns=df.iloc[:, 2:].values,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=SEED,
    )
    with open(SPLIT_CACHE, "wb") as f:
        pickle.dump((train_idx, val_idx, test_idx), f)
    print(f"Computed a fresh split and cached it to '{SPLIT_CACHE}'.")

df_graph_train = []
df_graph_val = []
df_graph_test = []
labels_train = []
labels_val = []
labels_test = []
G = []

for i in range(len(train_data)):
  # smiles = train_data['nonStereoSMILES'].iloc[i]
  # data = from_smiles(smiles, with_hydrogen=0)
  # data.x = data.x.float()
  # data.edge_attr = data.edge_attr.float()
  # data.y = torch.tensor(train_data.iloc[i, 2:].to_numpy(dtype=float), dtype=torch.float)
  # df_graph_train.append(data)
  # G.append(to_networkx(data, to_undirected=True))
  # labels_train.append(train_data.iloc[i, 2:])

  smiles  = train_data['nonStereoSMILES'].iloc[i]
  y       = torch.tensor(train_data.iloc[i, 2:].to_numpy(dtype=float), dtype=torch.float)
  data    = smiles_to_graph(smiles, y)

  if data is None:                        # RDKit failed to parse SMILES
      skipped.append(i)
      continue

  df_graph_train.append(data)
  # G.append(to_networkx(data, to_undirected=True))
  labels_train.append(train_data.iloc[i, 2:])

for i in range(len(val_data)):
  # smiles = val_data['nonStereoSMILES'].iloc[i]
  # data = from_smiles(smiles, with_hydrogen=0)
  # data.x = data.x.float()
  # data.edge_attr = data.edge_attr.float()
  # data.y = torch.tensor(val_data.iloc[i, 2:].to_numpy(dtype=float), dtype=torch.float)
  # df_graph_val.append(data)
  # G.append(to_networkx(data, to_undirected=True))
  # labels_val.append(val_data.iloc[i, 2:])

  smiles  = val_data['nonStereoSMILES'].iloc[i]
  y       = torch.tensor(val_data.iloc[i, 2:].to_numpy(dtype=float), dtype=torch.float)
  data    = smiles_to_graph(smiles, y)

  if data is None:                        # RDKit failed to parse SMILES
      skipped.append(i)
      continue

  df_graph_val.append(data)
  # G.append(to_networkx(data, to_undirected=True))
  labels_val.append(val_data.iloc[i, 2:])  

for i in range(len(test_data)):
  # smiles = test_data['nonStereoSMILES'].iloc[i]
  # data = from_smiles(smiles, with_hydrogen=0)
  # data.x = data.x.float()
  # data.edge_attr = data.edge_attr.float()
  # data.y = torch.tensor(test_data.iloc[i, 2:].to_numpy(dtype=float), dtype=torch.float)
  # df_graph_test.append(data)
  # G.append(to_networkx(data, to_undirected=True))
  # labels_test.append(test_data.iloc[i, 2:])

  smiles  = test_data['nonStereoSMILES'].iloc[i]
  y       = torch.tensor(test_data.iloc[i, 2:].to_numpy(dtype=float), dtype=torch.float)
  data    = smiles_to_graph(smiles, y)

  if data is None:                        # RDKit failed to parse SMILES
      skipped.append(i)
      continue

  df_graph_test.append(data)
  labels_test.append(test_data.iloc[i, 2:])

train_data = df_graph_train
val_data = df_graph_val
test_data = df_graph_test

train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)
val_loader   = DataLoader(val_data,   batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS) # no shuffle needed for eval
test_loader  = DataLoader(test_data,  batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)

# ## ________________________________________________________________________________________________________________________________________________________


# ## Training using the parameters of the models used in found papers
# 


# ## Paper 1:
# 
# >"Machine Learning for Scent: Learning Generalizable Perceptual Representations of Small Molecules"
# 
# Available on: https://arxiv.org/pdf/1910.10685
# 
# With the HMCN-F architecture implemented into the one in the paper above from the paper:
# 
# > "Hierarchical Multi-Label Classification Networks"
# 
# Available on: https://proceedings.mlr.press/v80/wehrmann18a/wehrmann18a.pdf


from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import numpy as np
import torch

def calculate_all_metrics(loader, model, device, criterion, threshold=0.5):
    model.eval()
    total_loss = 0
    y_true_all = []
    y_probs_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            logits_global, logits_local1, logits_local2 = model(batch)

            beta = model.beta
            out = beta * logits_local2 + (1 - beta) * logits_global

            y = batch.y.view(batch.num_graphs, -1).float()
            loss = criterion(logits_global, logits_local1, logits_local2, y)

            total_loss += loss.item()

            probs = torch.sigmoid(out)

            preds = (probs > threshold).float()

            y_true_all.append(batch.y.view(batch.num_graphs, -1).float().cpu())
            y_probs_all.append(probs.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())

    y_true = np.vstack(y_true_all)
    y_probs = np.vstack(y_probs_all)
    y_pred = np.vstack(y_pred_all)

    avg_loss = total_loss / len(loader)

    auroc = roc_auc_score(y_true, y_probs, average='micro')

    aucpr = average_precision_score(y_true, y_probs, average='micro')

    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return avg_loss, auroc, aucpr, precision, recall, f1, f1_macro

# ### Model


import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch.nn import Linear, BatchNorm1d, Dropout

class SmellGATV2_HMCNF(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_heads = 1, num_classes=138, num_parents=12, beta=0.5, dims=dims):
        super(SmellGATV2_HMCNF, self).__init__()

        c1, c2, c3, c4, g1, g2, l1t, l2t, drop = dims

        self.beta = beta
        self.num_classes = num_classes
        self.num_parents = num_parents

        self.conv1 = GATv2Conv(num_node_features, c1, edge_dim=num_edge_features, heads=num_heads, concat=False)
        self.conv2 = GATv2Conv(c1, c2, edge_dim=num_edge_features, heads=num_heads, concat=False)
        self.conv3 = GATv2Conv(c2, c3, edge_dim=num_edge_features, heads=num_heads, concat=False)
        self.conv4 = GATv2Conv(c3, c4, edge_dim=num_edge_features, heads=num_heads, concat=False)

        self.mlp_input_dim = num_node_features + sum(dims[:4])  # 98 + node_features

        self.global_mlp1 = Linear(self.mlp_input_dim, g1)
        self.global_bn1  = BatchNorm1d(g1)
        self.global_drop1 = Dropout(drop)

        self.global_mlp2 = Linear(g1, g2)
        self.global_bn2  = BatchNorm1d(g2)
        self.global_drop2 = Dropout(drop)

        self.global_out = Linear(g2, num_classes)

        self.local1_transition = Linear(g1, l1t)
        self.local1_bn         = BatchNorm1d(l1t)
        self.local1_out        = Linear(l1t, num_parents)

        self.local2_transition = Linear(g2, l2t)
        self.local2_bn         = BatchNorm1d(l2t)
        self.local2_out        = Linear(l2t, num_classes)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        x0 = x
        x1 = F.selu(self.conv1(x0, edge_index, edge_attr=edge_attr))
        x2 = F.selu(self.conv2(x1, edge_index, edge_attr=edge_attr))
        x3 = F.selu(self.conv3(x2, edge_index, edge_attr=edge_attr))
        x4 = F.selu(self.conv4(x3, edge_index, edge_attr=edge_attr))

        g0 = global_add_pool(x0, batch)
        g1 = global_add_pool(x1, batch)
        g2 = global_add_pool(x2, batch)
        g3 = global_add_pool(x3, batch)
        g4 = global_add_pool(x4, batch)

        graph_repr = torch.cat([g0, g1, g2, g3, g4], dim=1)

        h1 = self.global_mlp1(graph_repr)
        h1 = self.global_bn1(h1)
        h1 = F.relu(h1)
        h1 = self.global_drop1(h1)

        h2 = self.global_mlp2(h1)
        h2 = self.global_bn2(h2)
        h2 = F.relu(h2)
        h2 = self.global_drop2(h2)

        logits_global = self.global_out(h2)

        l1 = F.relu(self.local1_bn(self.local1_transition(h1)))
        logits_local1 = self.local1_out(l1)

        l2 = F.relu(self.local2_bn(self.local2_transition(h2)))
        logits_local2 = self.local2_out(l2)

        return logits_global, logits_local1, logits_local2

# ### Building the parent ground truth


import torch

def get_parent_labels(y_138, child_parent_pairs, num_parents=12):
    batch_size = y_138.shape[0]

    parent_tensors = [[] for _ in range(num_parents)]
    for child_idx, parent_idx in child_parent_pairs:
        parent_tensors[parent_idx].append(y_138[:, child_idx])

    cols = []
    for group in parent_tensors:
        if group:
            cols.append(torch.stack(group, dim=0).max(dim=0).values)
        else:
            cols.append(torch.zeros(batch_size, device=y_138.device))

    return torch.stack(cols, dim=1)

# ### Hierarchical Loss Function


child_idxs  = [cp[0] for cp in child_parent_pairs]
parent_idxs = [cp[1] for cp in child_parent_pairs]

def hmcnf_loss(logits_global, logits_local1, logits_local2, child_idxs, parent_idxs,
               y_138, child_parent_pairs,
               pos_weight=None, lambda_hier=0.3):

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Derive y_parents from y_138 (OR aggregation over children)
    y_parents = get_parent_labels(y_138, child_parent_pairs, num_parents=12)

    # L_global: BCE on all 138 labels
    loss_global = bce(logits_global, y_138)

    # L_local1: BCE on 12 parent groups
    bce_unweighted = torch.nn.BCEWithLogitsLoss()
    loss_local1 = bce_unweighted(logits_local1, y_parents)

    # L_local2: BCE on 138 fine-grained labels
    loss_local2 = bce(logits_local2, y_138)

    # L_hierarchy: squared violation penalty on validated pairs (paper uses squared)
    probs_global = torch.sigmoid(logits_global)
    probs_parent = get_parent_labels(probs_global, child_parent_pairs, num_parents=12)

    penalty = torch.tensor(0.0, device=logits_global.device)
    # for child_idx, parent_idx in child_parent_pairs:
    #     p_child  = probs_global[:, child_idx]
    #     p_parent = probs_parent[:, parent_idx]
    #     penalty  = penalty + (torch.relu(p_child - p_parent) ** 2).mean()

    # child_idxs  = [cp[0] for cp in child_parent_pairs]
    # parent_idxs = [cp[1] for cp in child_parent_pairs]

    p_children = probs_global[:, child_idxs]   # (batch, N_pairs)
    p_parents  = probs_parent[:, parent_idxs]  # (batch, N_pairs)
    penalty    = (torch.relu(p_children - p_parents) ** 2).mean()

    penalty = penalty / len(child_parent_pairs)

    return loss_global + loss_local1 + loss_local2 + lambda_hier * penalty

# ### Training


# import torch
# import copy
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# number_of_classes = (df.shape[1] - 2)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print(f"Memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

# model = SmellGATV2_HMCNF(
#     num_node_features=df_graph_train[0].x.shape[1],
#     num_edge_features=df_graph_train[0].edge_attr.shape[1],
#     num_heads=num_heads,
#     num_classes=138,
#     num_parents=12,
#     beta=0.5
# ).to(device)

# all_y_tensors = [data.y for data in train_data]
# stacked_y_tensors = torch.stack(all_y_tensors)
# num_positives = torch.sum(stacked_y_tensors, dim=0)
# num_negatives = len(train_data) - num_positives
# pos_weight = num_negatives / (num_positives + 1e-5)
# pos_weight = pos_weight.to(device)

# best_val_auroc = 0.0
# best_model_state = None

# lambda_hier = lambda_hier # new parameter to adjust for the HMC to work properly
# epochs = epochs
# best_epoch = 0

# # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# criterion = lambda lg, ll1, ll2, y: hmcnf_loss(
#     lg, ll1, ll2, child_idxs, parent_idxs, y, child_parent_pairs,
#     pos_weight=pos_weight, lambda_hier=lambda_hier
# )
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0_cosine, T_mult=1)

# history = {
#     'train_loss': [],
#     'val_loss': [],
#     'val_auroc': [],
#     'val_aucpr': [],
#     'val_precision': [],
#     'val_recall': [],
#     'val_f1': [],
#     'val_f1_macro': []
# }

# print(f"Starting Training ({epochs} epochs)...")

# for epoch in range(epochs):
#   model.train()
#   train_loss = 0

#   for batch in train_loader:
#     batch = batch.to(device)
#     optimizer.zero_grad()

#     logits_global, logits_local1, logits_local2 = model(batch)  # unpack 3 outputs
#     y = batch.y.view(batch.num_graphs, -1).float()

#     loss = criterion(logits_global, logits_local1, logits_local2, y)
#     loss.backward()
#     optimizer.step()
#     train_loss += loss.item()

#   scheduler.step()
#   avg_train_loss = train_loss / len(train_loader)

#   val_loss, val_auroc, val_aucpr, val_prec, val_rec, val_f1, val_f1_macro = calculate_all_metrics(val_loader, model, device, criterion)

#   # Save best model
#   if val_auroc > best_val_auroc:
#       best_val_auroc = val_auroc
#       best_model_state = copy.deepcopy(model.state_dict())
#       best_epoch = epoch

#   # Store
#   history['train_loss'].append(avg_train_loss)
#   history['val_loss'].append(val_loss)
#   history['val_auroc'].append(val_auroc)
#   history['val_aucpr'].append(val_aucpr)
#   history['val_precision'].append(val_prec)
#   history['val_recall'].append(val_rec)
#   history['val_f1'].append(val_f1)
#   history['val_f1_macro'].append(val_f1_macro)

#   if (epoch + 1) % 10 == 0:
#       print(f"Epoch {epoch+1:03d} | Loss: {val_loss:.4f} | F1: {val_f1:.4f} | AUROC: {val_auroc:.4f}")

# print("Training Complete.")

# print(f"\n--- Best Epoch: {best_epoch + 1} ---")
# print(f"  Lambda     : {lambda_hier:.4f}")
# print(f"  Train Loss : {history['train_loss'][best_epoch]:.4f}")
# print(f"  Val Loss   : {history['val_loss'][best_epoch]:.4f}")
# print(f"  AUROC      : {history['val_auroc'][best_epoch]:.4f}")
# print(f"  AUCPR      : {history['val_aucpr'][best_epoch]:.4f}")
# print(f"  Precision  : {history['val_precision'][best_epoch]:.4f}")
# print(f"  Recall     : {history['val_recall'][best_epoch]:.4f}")
# print(f"  F1 (micro) : {history['val_f1'][best_epoch]:.4f}")
# print(f"  F1 (macro) : {history['val_f1_macro'][best_epoch]:.4f}")

# model.load_state_dict(best_model_state)

# ### Optimizing the threshold before testing (using the validation set for it)
#   - Using one threshold adjusted for each label (138)


import numpy as np
from sklearn.metrics import f1_score
import torch

def find_per_label_thresholds(val_loader, model, device, num_classes=138):
    model.eval()
    y_true_all = []
    y_probs_all = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            logits_global, logits_local1, logits_local2 = model(batch)
            beta = model.beta
            out = beta * logits_local2 + (1 - beta) * logits_global

            probs = torch.sigmoid(out)

            y_true_all.append(batch.y.view(batch.num_graphs, -1).float().cpu())
            y_probs_all.append(probs.cpu().numpy())

    y_true = np.vstack(y_true_all)
    y_probs = np.vstack(y_probs_all)

    best_thresholds = np.full(num_classes, 0.5)

    print(f"Sweeping thresholds for all {num_classes} labels individually...")

    for class_idx in range(num_classes):
        y_true_class = y_true[:, class_idx]
        y_probs_class = y_probs[:, class_idx]

        if np.sum(y_true_class) == 0:
            continue

        best_f1 = 0.0
        best_thresh = 0.5

        for thresh in np.arange(0.01, 1.0, 0.01):
            y_pred_class = (y_probs_class >= thresh).astype(int)
            current_f1 = f1_score(y_true_class, y_pred_class, average = 'macro', zero_division=0)

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_thresh = thresh

        best_thresholds[class_idx] = best_thresh

    print("Done! Found 138 optimal thresholds.")
    return best_thresholds

# - Using one threshold adjusted for each label (12)


def find_per_label_thresholds_12(val_loader, model, device, num_parents=12):
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits_global, logits_local1, logits_local2 = model(batch)

            # Use local1 output for 12-group predictions
            probs_12 = torch.sigmoid(logits_local1)

            # Derive ground truth 12-group labels from y_138
            y_138 = batch.y.view(batch.num_graphs, -1).float()
            y_12  = get_parent_labels(y_138, child_parent_pairs, num_parents=12)

            all_probs.append(probs_12.cpu())
            all_labels.append(y_12.cpu())

    all_probs  = torch.cat(all_probs,  dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Per-label threshold sweep — identical logic to your 138-label version
    thresholds_12 = []
    for i in range(num_parents):
        best_thresh, best_f1 = 0.5, 0.0
        for t in torch.arange(0.1, 0.9, 0.01):
            preds = (all_probs[:, i] >= t.item()).astype(int)
            f1 = f1_score(all_labels[:, i], preds, average = 'macro', zero_division=0)
            if f1 > best_f1:
                best_f1    = f1
                best_thresh = t.item()
        thresholds_12.append(best_thresh)

    return torch.tensor(thresholds_12)

# optimal_thresholds = find_per_label_thresholds(val_loader, model, device, num_classes = number_of_classes)
# thresholds_12  = find_per_label_thresholds_12(val_loader, model, device, num_parents=12)

# thresholds_tensor = torch.tensor(optimal_thresholds, dtype=torch.float32).to(device)
# thresholds_12 = torch.tensor(thresholds_12, dtype=torch.float32).to(device)

import pandas as pd
import os

def save_metrics_to_csv(csv_path, **kwargs):
    """Appends a single row of metrics to a CSV, creating it if it doesn't exist."""
    row = pd.DataFrame([kwargs])
    if os.path.exists(csv_path):
        row.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        row.to_csv(csv_path, mode='w', header=True, index=False)

from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              precision_score, recall_score, accuracy_score,
                              balanced_accuracy_score)
import numpy as np

def calculate_all_metrics_thresh(loader, model, device, criterion,
                                  thresholds_138, thresholds_12):
    model.eval()
    total_loss = 0

    y_true_138_all  = []
    y_probs_138_all = []

    y_true_12_all   = []
    y_probs_12_all  = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            logits_global, logits_local1, logits_local2 = model(batch)
            beta = model.beta
            out  = beta * logits_local2 + (1 - beta) * logits_global

            y_138 = batch.y.view(batch.num_graphs, -1).float()
            loss  = criterion(logits_global, logits_local1, logits_local2, y_138)
            total_loss += loss.item()

            probs_138 = torch.sigmoid(out)

            probs_12  = torch.sigmoid(logits_local1)
            y_12      = get_parent_labels(y_138, child_parent_pairs, num_parents=12)

            y_true_138_all.append(y_138.cpu().numpy())
            y_probs_138_all.append(probs_138.cpu().numpy())
            y_true_12_all.append(y_12.cpu().numpy())
            y_probs_12_all.append(probs_12.cpu().numpy())

    avg_loss   = total_loss / len(loader)

    y_true_138 = np.vstack(y_true_138_all)
    y_probs_138 = np.vstack(y_probs_138_all)
    y_true_12  = np.vstack(y_true_12_all)
    y_probs_12 = np.vstack(y_probs_12_all)

    y_pred_138 = (y_probs_138 >= np.array(thresholds_138)).astype(int)
    y_pred_12  = (y_probs_12  >= np.array(thresholds_12.cpu())).astype(int)


    valid_cols_138 = [i for i in range(138)
                      if len(np.unique(y_true_138[:, i])) > 1]

    auroc_138 = roc_auc_score(y_true_138[:, valid_cols_138],
                               y_probs_138[:, valid_cols_138],
                               average='macro')
    aucpr_138 = average_precision_score(y_true_138[:, valid_cols_138],
                                         y_probs_138[:, valid_cols_138],
                                         average='macro')
    f1_138    = f1_score(y_true_138, y_pred_138, average='macro', zero_division=0)

    violations, total_child_pos = 0, 0
    for child_idx, parent_idx in child_parent_pairs:
        child_pred  = y_pred_138[:, child_idx]
        parent_pred = y_pred_12[:, parent_idx]
        violations      += ((child_pred == 1) & (parent_pred == 0)).sum()
        total_child_pos += (child_pred == 1).sum()
    hier_violation_rate = violations / (total_child_pos + 1e-8)

    valid_cols_12 = [i for i in range(12)
                     if len(np.unique(y_true_12[:, i])) > 1]

    auroc_12 = roc_auc_score(y_true_12[:, valid_cols_12],
                              y_probs_12[:, valid_cols_12],
                              average='macro')
    aucpr_12 = average_precision_score(y_true_12[:, valid_cols_12],
                                        y_probs_12[:, valid_cols_12],
                                        average='macro')
    f1_12    = f1_score(y_true_12, y_pred_12, average='macro', zero_division=0)

    bal_acc_scores, sens_scores, spec_scores = [], [], []
    for i in range(12):
        if len(np.unique(y_true_12[:, i])) < 2:
            continue
        tn = ((y_pred_12[:, i] == 0) & (y_true_12[:, i] == 0)).sum()
        fp = ((y_pred_12[:, i] == 1) & (y_true_12[:, i] == 0)).sum()
        fn = ((y_pred_12[:, i] == 0) & (y_true_12[:, i] == 1)).sum()
        tp = ((y_pred_12[:, i] == 1) & (y_true_12[:, i] == 1)).sum()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        sens_scores.append(sens)
        spec_scores.append(spec)
        bal_acc_scores.append((sens + spec) / 2)

    bal_acc_12  = np.mean(bal_acc_scores)
    sensitivity = np.mean(sens_scores)
    specificity = np.mean(spec_scores)

    cooc_true  = (y_true_12.T @ y_true_12)
    cooc_pred  = (y_pred_12.T @ y_pred_12)
    cooc_true_norm = cooc_true / (cooc_true.max() + 1e-8)
    cooc_pred_norm = cooc_pred / (cooc_pred.max() + 1e-8)
    label_cooc_consistency = np.mean(np.abs(cooc_true_norm - cooc_pred_norm))

    w = 62
    print("-" * w)
    print("Setup:\n")
    print(f"  {'Number of Epochs':<30}: {epochs}\n")
    print(f"  {'Best Epoch':<30}: {best_epoch + 1}\n")
    print(f"  {'Lambda':<30}: {lambda_hier:.4f}\n")
    print(f"  {'Learning Rate':<30}: {learning_rate:.4f}\n")
    print(f"  {'T_0 (Cosine Restarts)':<30}: {t_0_cosine}\n")
    print(f"  {'Batch Size':<30}: {batch_size}\n")
    print(f"  {'Num Heads':<30}: {num_heads}\n")
    print("-" * w)
    print(f"  HMCN Fine — 138 labels  ({len(valid_cols_138)}/138 used for AUC)\n")
    print("-" * w)
    print(f"  {'ROC AUC':<30}: {auroc_138:.4f}\n")
    print(f"  {'PR AUC':<30}: {aucpr_138:.4f}\n")
    print(f"  {'F1 (macro)':<30}: "
          f"{f1_score(y_true_138, y_pred_138, average='macro', zero_division=0):.4f}\n")
    print(f"  {'F1 (micro)':<30}: "
          f"{f1_score(y_true_138, y_pred_138, average='micro', zero_division=0):.4f}\n")
    print(f"  {'Hierarchical Violation Rate':<30}: {hier_violation_rate:.4f}\n")
    print(f"  {'Precision (micro)':<30}: {precision_score(y_true_138, y_pred_138, average='micro', zero_division=0):.4f}\n")
    print(f"  {'Recall (micro)':<30}: {recall_score(y_true_138, y_pred_138, average='micro', zero_division=0):.4f}\n")
    print(f"  {'Precision (macro)':<30}: {precision_score(y_true_138, y_pred_138, average='macro', zero_division=0):.4f}\n")
    print(f"  {'Recall (macro)':<30}: {recall_score(y_true_138, y_pred_138, average='macro', zero_division=0):.4f}\n")
    # print(f"  {'Balanced Accuracy':<30}: {balanced_accuracy_score(y_true_138, y_pred_138):.4f}\n")
    print(f"  {'Accuracy':<30}: {accuracy_score(y_true_138, y_pred_138):.4f}\n")
    print("-" * w)
    print(f"  HMCN Meta — 12 groups   ({len(valid_cols_12)}/12 used for AUC)\n")
    print("-" * w)
    print(f"  {'ROC AUC':<30}: {auroc_12:.4f}\n")
    print(f"  {'PR AUC':<30}: {aucpr_12:.4f}\n")
    print(f"  {'F1 (macro)':<30}: {f1_12:.4f}\n")
    print(f"  {'Instance-F1':<30}: "
          f"{f1_score(y_true_12, y_pred_12, average='samples', zero_division=0):.4f}\n")
    print(f"  {'Balanced Accuracy':<30}: {bal_acc_12:.4f}\n")
    print(f"  {'Accuracy':<30}: {accuracy_score(y_true_12, y_pred_12):.4f}\n")
    print(f"  {'Sensitivity (macro)':<30}: {sensitivity:.4f}\n")
    print(f"  {'Specificity (macro)':<30}: {specificity:.4f}\n")
    print(f"  {'Hierarchical Violation Rate':<30}: {hier_violation_rate:.4f}\n")
    print(f"  {'Label Co-occurrence Consistency':<30}: {label_cooc_consistency:.4f}\n")
    print("-" * w)
    print(f"  Loss: {avg_loss:.4f}\n")
    print("-" * w)

    save_metrics_to_csv(
        csv_path=metrics_path,

        epochs=epochs,
        best_epoch=best_epoch + 1,
        lambda_hier=lambda_hier,
        learning_rate=learning_rate,
        t_0_cosine=t_0_cosine,
        batch_size=batch_size,
        num_heads=num_heads,

        roc_auc_138=auroc_138,
        pr_auc_138=aucpr_138,
        f1_macro_138=f1_score(y_true_138, y_pred_138, average='macro', zero_division=0),
        f1_micro_138=f1_score(y_true_138, y_pred_138, average='micro', zero_division=0),
        hier_violation_rate_138=hier_violation_rate,
        precision_micro_138=precision_score(y_true_138, y_pred_138, average='micro', zero_division=0),
        recall_micro_138=recall_score(y_true_138, y_pred_138, average='micro', zero_division=0),
        precision_macro_138=precision_score(y_true_138, y_pred_138, average='macro', zero_division=0),
        recall_macro_138=recall_score(y_true_138, y_pred_138, average='macro', zero_division=0),
        accuracy_138=accuracy_score(y_true_138, y_pred_138),

        roc_auc_12=auroc_12,
        pr_auc_12=aucpr_12,
        f1_macro_12=f1_12,
        instance_f1_12=f1_score(y_true_12, y_pred_12, average='samples', zero_division=0),
        balanced_accuracy_12=bal_acc_12,
        accuracy_12=accuracy_score(y_true_12, y_pred_12),
        sensitivity_macro_12=sensitivity,
        specificity_macro_12=specificity,
        hier_violation_rate_12=hier_violation_rate,
        label_cooc_consistency_12=label_cooc_consistency,

        loss=avg_loss,
    )

# ### Sweep


import torch
import copy
import os
import csv
import itertools
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from datetime import datetime

def run_sweep(
    # ── Data (already built, passed in) ──────────────────────────────────────
    train_data, val_data,
    df_graph_train,
    child_parent_pairs, child_idxs, parent_idxs,
    num_node_features, num_edge_features,
    # ── Sweep grid ───────────────────────────────────────────────────────────
    lambda_values     = [0.0, 0.01, 0.05, 0.1],
    heads_values      = [1, 4, 8],
    t0_values         = [50, 100, 200],   # CosineAnnealingWarmRestarts T_0
    epochs_values     = [500, 1000],
    # ── Fixed hyperparameters ─────────────────────────────────────────────────
    beta              = 0.5,
    dropout           = 0.47,
    learning_rate     = 1e-3,
    batch_size        = 128,
    num_classes       = 138,
    num_parents       = 12,
    # ── Output ───────────────────────────────────────────────────────────────
    results_csv       = "sweep_results.csv",
    checkpoint_dir    = "checkpoints",
    monitor           = "val_auroc",   # "val_auroc" | "val_loss" | "val_f1_macro"
    device            = None,
):
    """
    Grid-sweep over (lambda_hier × num_heads × T_0 × epochs) for SmellGATV2_HMCNF.

    For each combination the function:
      1. Builds a fresh model + optimiser + scheduler.
      2. Trains for `epochs` epochs, checkpointing the best epoch by `monitor`.
      3. Evaluates the best checkpoint on the validation set.
      4. Appends one row to `results_csv` (safe to interrupt and resume).

    Parameters
    ----------
    monitor : str
        Which val metric to use for best-checkpoint selection.
        "val_auroc"   → maximise AUROC   (best for ranking quality)
        "val_loss"    → minimise loss    (most stable signal)
        "val_f1_macro"→ maximise F1-macro (best for per-label quality)
    """
    from torch_geometric.data import DataLoader

    gpu = '0'

    if device is None:
        device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    print(f"Sweep device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── pos_weight (computed once, reused for every run) ──────────────────────
    all_y = torch.stack([d.y for d in train_data])
    num_pos = all_y.sum(0)
    num_neg = len(train_data) - num_pos
    pos_weight = (num_neg / (num_pos + 1e-5)).to(device)

    # ── DataLoaders (rebuilt once; same across all runs) ──────────────────────
    set_seed(SEED)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)

    # ── CSV header ────────────────────────────────────────────────────────────
    fieldnames = [
        "run_id", "timestamp",
        "lambda_hier", "num_heads", "T_0", "epochs",
        "beta", "dropout", "lr",
        "best_epoch", "monitor_metric",
        "val_loss", "val_auroc", "val_aucpr",
        "val_precision", "val_recall", "val_f1_micro", "val_f1_macro",
    ]
    write_header = not os.path.exists(results_csv)
    csv_file = open(results_csv, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    # ── Helper: build a fresh criterion (captures lambda in closure) ──────────
    def make_criterion(lambda_hier_val):
        return lambda lg, ll1, ll2, y: hmcnf_loss(
            lg, ll1, ll2, child_idxs, parent_idxs, y, child_parent_pairs,
            pos_weight=pos_weight, lambda_hier=lambda_hier_val,
        )

    # ── Helper: pick "better" according to monitor ────────────────────────────
    def is_better(new_val, best_val):
        if monitor == "val_loss":
            return new_val < best_val
        return new_val > best_val  # auroc / f1_macro / aucpr → higher is better

    best_init = float("inf") if monitor == "val_loss" else 0.0

    # ── Main sweep loop ───────────────────────────────────────────────────────
    grid = list(itertools.product(lambda_values, heads_values, t0_values, epochs_values))
    print(f"\nTotal runs: {len(grid)}\n{'='*60}")

    for run_idx, (lam, heads, t0, epochs) in enumerate(grid):
        run_id  = f"run{run_idx:03d}_lam{lam}_h{heads}_t0{t0}_ep{epochs}"
        ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{run_idx+1}/{len(grid)}] {run_id}  started {ts_start}")

        # ── Reset RNG state so every run starts from the same initial weights ──
        set_seed(SEED)

        # ── Fresh model ───────────────────────────────────────────────────────
        model = SmellGATV2_HMCNF(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_heads=heads,
            num_classes=num_classes,
            num_parents=num_parents,
            beta=beta,
        ).to(device)

        # Patch dropout if you want to vary it later:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = dropout

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=1)
        criterion = make_criterion(lam)

        best_metric   = best_init
        best_state    = None
        best_epoch    = 0
        history_auroc = []   # lightweight log, no need to store everything

        # ── Training loop ─────────────────────────────────────────────────────
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                lg, ll1, ll2 = model(batch)
                y = batch.y.view(batch.num_graphs, -1).float()
                loss = criterion(lg, ll1, ll2, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            val_loss, val_auroc, val_aucpr, val_prec, val_rec, val_f1, val_f1_macro = \
                calculate_all_metrics(val_loader, model, device, criterion)

            # Monitor-specific value for checkpointing
            monitor_val = {
                "val_auroc":    val_auroc,
                "val_loss":     val_loss,
                "val_f1_macro": val_f1_macro,
            }[monitor]

            if is_better(monitor_val, best_metric):
                best_metric = monitor_val
                best_state  = copy.deepcopy(model.state_dict())
                best_epoch  = epoch + 1
                best_metrics_snapshot = dict(
                    val_loss=val_loss, val_auroc=val_auroc, val_aucpr=val_aucpr,
                    val_precision=val_prec, val_recall=val_rec,
                    val_f1_micro=val_f1, val_f1_macro=val_f1_macro,
                )

            if (epoch + 1) % 5 == 0:
                print(f"  ep {epoch+1:4d} | loss {val_loss:.4f} | "
                      f"auroc {val_auroc:.4f} | f1_macro {val_f1_macro:.4f}")

        # ── Save best checkpoint ──────────────────────────────────────────────
        ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_best.pt")
        torch.save(best_state, ckpt_path)

        # ── Log to CSV ────────────────────────────────────────────────────────
        row = dict(
            run_id=run_id, timestamp=ts_start,
            lambda_hier=lam, num_heads=heads, T_0=t0, epochs=epochs,
            beta=beta, dropout=dropout, lr=learning_rate,
            best_epoch=best_epoch, monitor_metric=monitor,
            **best_metrics_snapshot,
        )
        writer.writerow(row)
        csv_file.flush()

        print(f"  → Best epoch {best_epoch} | auroc {best_metrics_snapshot['val_auroc']:.4f} "
              f"| f1_macro {best_metrics_snapshot['val_f1_macro']:.4f} "
              f"| aucpr {best_metrics_snapshot['val_aucpr']:.4f}")
        print(f"  → Saved to {ckpt_path}")

    csv_file.close()
    print(f"\n{'='*60}\nSweep complete. Results saved to '{results_csv}'.")

import os
import csv
import torch
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    accuracy_score, balanced_accuracy_score,
    jaccard_score, hamming_loss,
)

def evaluate_sweep(
    # ── Sweep artefacts ───────────────────────────────────────────────────────
    sweep_csv          = "sweep_results.csv",
    checkpoint_dir     = "checkpoints",
    # ── Data ─────────────────────────────────────────────────────────────────
    val_data           = None,
    test_data          = None,
    child_parent_pairs = None,
    num_node_features  = None,
    num_edge_features  = None,
    # ── Fixed architectural params (must match run_sweep) ─────────────────────
    num_classes        = 138,
    num_parents        = 12,
    # ── Output ───────────────────────────────────────────────────────────────
    output_csv         = "sweep_test_results.csv",
    batch_size         = 128,
    device             = None,
):
    """
    For every checkpoint saved by run_sweep():
      1. Loads the model weights.
      2. Calibrates per-label thresholds on the validation set (138 + 12).
      3. Evaluates on the test set with those thresholds.
      4. Writes one row to output_csv with all hyperparameters + all test metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluate device: {device}")

    val_loader  = DataLoader(val_data,  batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)

    child_idxs  = [cp[0] for cp in child_parent_pairs]
    parent_idxs = [cp[1] for cp in child_parent_pairs]

    # ── CSV columns ───────────────────────────────────────────────────────────
    hp_cols  = ["run_id", "timestamp",
                "lambda_hier", "num_heads", "T_0", "epochs",
                "beta", "dropout", "lr",
                "best_epoch", "monitor_metric"]

    val_cols = ["val_loss", "val_auroc", "val_aucpr",
                "val_precision", "val_recall", "val_f1_micro", "val_f1_macro"]

    test_cols_138 = [
        "test_auroc_138", "test_aucpr_138",
        "test_f1_macro_138", "test_f1_micro_138",
        "test_hier_violation_rate_138",
        "test_precision_micro_138", "test_recall_micro_138",
        "test_precision_macro_138", "test_recall_macro_138",
        "test_accuracy_138",
        "test_jaccard_macro_138", "test_jaccard_micro_138",
        "test_hamming_loss_138",
    ]
    test_cols_12 = [
        "test_auroc_12", "test_aucpr_12",
        "test_f1_macro_12", "test_f1_micro_12", "test_f1_instance_12",
        "test_balanced_accuracy_12",
        "test_accuracy_12",
        "test_sensitivity_macro_12", "test_specificity_macro_12",
        "test_hier_violation_rate_12",
        "test_label_cooc_consistency_12",
        "test_jaccard_macro_12", "test_jaccard_micro_12",
        "test_hamming_loss_12",
    ]
    misc_cols = ["test_loss"]

    fieldnames = hp_cols + val_cols + test_cols_138 + test_cols_12 + misc_cols

    write_header = not os.path.exists(output_csv)
    out_file = open(output_csv, "a", newline="")
    writer   = csv.DictWriter(out_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        out_file.flush()

    # ── Read sweep CSV ────────────────────────────────────────────────────────
    with open(sweep_csv, "r") as f:
        sweep_rows = list(csv.DictReader(f))
    print(f"Found {len(sweep_rows)} runs in '{sweep_csv}'.\n{'='*60}")

    for i, row in enumerate(sweep_rows):
        run_id = row["run_id"]
        print(f"\n[{i+1}/{len(sweep_rows)}] {run_id}")

        # ── Rebuild model ─────────────────────────────────────────────────────
        num_heads = int(row["num_heads"])
        beta      = float(row["beta"])

        model = SmellGATV2_HMCNF(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_heads=num_heads,
            num_classes=num_classes,
            num_parents=num_parents,
            beta=beta,
        ).to(device)

        ckpt_path = os.path.join(checkpoint_dir, f"{run_id}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found at {ckpt_path}, skipping.")
            continue
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        lambda_hier      = float(row["lambda_hier"])
        pos_weight_dummy = torch.ones(num_classes, device=device)
        criterion = lambda lg, ll1, ll2, y: hmcnf_loss(
            lg, ll1, ll2, child_idxs, parent_idxs, y, child_parent_pairs,
            pos_weight=pos_weight_dummy, lambda_hier=lambda_hier,
        )

        # ── Calibrate thresholds on val set ───────────────────────────────────
        print("  Calibrating thresholds on val set...")
        thresholds_138 = torch.tensor(
            find_per_label_thresholds(val_loader, model, device, num_classes=num_classes),
            dtype=torch.float32,
        ).to(device)
        thresholds_12 = find_per_label_thresholds_12(
            val_loader, model, device, num_parents=num_parents
        ).to(device)

        # ── Collect raw test-set predictions ──────────────────────────────────
        print("  Collecting test-set predictions...")
        model.eval()
        y_true_138_all, y_probs_138_all = [], []
        y_true_12_all,  y_probs_12_all  = [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits_global, logits_local1, logits_local2 = model(batch)

                # 138-label blended output (same as your calculate_all_metrics_thresh)
                out_138 = beta * logits_local2 + (1 - beta) * logits_global
                probs_138 = torch.sigmoid(out_138)

                # 12-group output
                probs_12 = torch.sigmoid(logits_local1)

                y_138 = batch.y.view(batch.num_graphs, -1).float()
                y_12  = get_parent_labels(y_138, child_parent_pairs, num_parents=num_parents)

                loss = criterion(logits_global, logits_local1, logits_local2, y_138)
                total_loss += loss.item()

                y_true_138_all.append(y_138.cpu())
                y_probs_138_all.append(probs_138.cpu().numpy())
                y_true_12_all.append(y_12.cpu())
                y_probs_12_all.append(probs_12.cpu().numpy())

        y_true_138 = np.vstack(y_true_138_all)
        y_probs_138 = np.vstack(y_probs_138_all)
        y_true_12  = np.vstack(y_true_12_all)
        y_probs_12  = np.vstack(y_probs_12_all)

        y_pred_138 = (y_probs_138 >= thresholds_138.cpu().numpy()).astype(int)
        y_pred_12  = (y_probs_12  >= thresholds_12.cpu().numpy()).astype(int)

        avg_loss = total_loss / len(test_loader)

        # ── 138-label metrics ─────────────────────────────────────────────────
        valid_cols_138 = [j for j in range(num_classes)
                          if len(np.unique(y_true_138[:, j])) > 1]

        auroc_138 = roc_auc_score(
            y_true_138[:, valid_cols_138], y_probs_138[:, valid_cols_138], average='macro')
        aucpr_138 = average_precision_score(
            y_true_138[:, valid_cols_138], y_probs_138[:, valid_cols_138], average='macro')

        # hierarchical violation rate: child predicted positive but parent not
        violations, total_child_pos = 0, 0
        for child_idx, parent_idx in child_parent_pairs:
            child_pred  = y_pred_138[:, child_idx]
            parent_pred = y_pred_12[:, parent_idx]
            violations      += ((child_pred == 1) & (parent_pred == 0)).sum()
            total_child_pos += (child_pred == 1).sum()
        hier_violation_rate = violations / (total_child_pos + 1e-8)

        # ── 12-group metrics ──────────────────────────────────────────────────
        valid_cols_12 = [j for j in range(num_parents)
                         if len(np.unique(y_true_12[:, j])) > 1]

        auroc_12 = roc_auc_score(
            y_true_12[:, valid_cols_12], y_probs_12[:, valid_cols_12], average='macro')
        aucpr_12 = average_precision_score(
            y_true_12[:, valid_cols_12], y_probs_12[:, valid_cols_12], average='macro')

        bal_acc_scores, sens_scores, spec_scores = [], [], []
        for j in range(num_parents):
            if len(np.unique(y_true_12[:, j])) < 2:
                continue
            tn = ((y_pred_12[:, j] == 0) & (y_true_12[:, j] == 0)).sum()
            fp = ((y_pred_12[:, j] == 1) & (y_true_12[:, j] == 0)).sum()
            fn = ((y_pred_12[:, j] == 0) & (y_true_12[:, j] == 1)).sum()
            tp = ((y_pred_12[:, j] == 1) & (y_true_12[:, j] == 1)).sum()
            sens = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
            sens_scores.append(sens)
            spec_scores.append(spec)
            bal_acc_scores.append((sens + spec) / 2)

        bal_acc_12  = np.mean(bal_acc_scores)
        sensitivity = np.mean(sens_scores)
        specificity = np.mean(spec_scores)

        cooc_true = (y_true_12.T @ y_true_12)
        cooc_pred = (y_pred_12.T @ y_pred_12)
        cooc_true_norm = cooc_true / (cooc_true.max() + 1e-8)
        cooc_pred_norm = cooc_pred / (cooc_pred.max() + 1e-8)
        label_cooc_consistency = np.mean(np.abs(cooc_true_norm - cooc_pred_norm))

        # ── Write row ─────────────────────────────────────────────────────────
        out_row = {col: row.get(col, "") for col in hp_cols + val_cols}
        out_row.update({
            # 138 labels
            "test_auroc_138":               f"{auroc_138:.4f}",
            "test_aucpr_138":               f"{aucpr_138:.4f}",
            "test_f1_macro_138":            f"{f1_score(y_true_138, y_pred_138, average='macro', zero_division=0):.4f}",
            "test_f1_micro_138":            f"{f1_score(y_true_138, y_pred_138, average='micro', zero_division=0):.4f}",
            "test_hier_violation_rate_138": f"{hier_violation_rate:.4f}",
            "test_precision_micro_138":     f"{precision_score(y_true_138, y_pred_138, average='micro', zero_division=0):.4f}",
            "test_recall_micro_138":        f"{recall_score(y_true_138, y_pred_138, average='micro', zero_division=0):.4f}",
            "test_precision_macro_138":     f"{precision_score(y_true_138, y_pred_138, average='macro', zero_division=0):.4f}",
            "test_recall_macro_138":        f"{recall_score(y_true_138, y_pred_138, average='macro', zero_division=0):.4f}",
            "test_accuracy_138":            f"{accuracy_score(y_true_138, y_pred_138):.4f}",
            "test_jaccard_macro_138":       f"{jaccard_score(y_true_138, y_pred_138, average='macro', zero_division=0):.4f}",
            "test_jaccard_micro_138":       f"{jaccard_score(y_true_138, y_pred_138, average='micro', zero_division=0):.4f}",
            "test_hamming_loss_138":        f"{hamming_loss(y_true_138, y_pred_138):.4f}",
            # 12 groups
            "test_auroc_12":                f"{auroc_12:.4f}",
            "test_aucpr_12":                f"{aucpr_12:.4f}",
            "test_f1_macro_12":             f"{f1_score(y_true_12, y_pred_12, average='macro', zero_division=0):.4f}",
            "test_f1_micro_12":             f"{f1_score(y_true_12, y_pred_12, average='micro', zero_division=0):.4f}",
            "test_f1_instance_12":          f"{f1_score(y_true_12, y_pred_12, average='samples', zero_division=0):.4f}",
            "test_balanced_accuracy_12":    f"{bal_acc_12:.4f}",
            "test_accuracy_12":             f"{accuracy_score(y_true_12, y_pred_12):.4f}",
            "test_sensitivity_macro_12":    f"{sensitivity:.4f}",
            "test_specificity_macro_12":    f"{specificity:.4f}",
            "test_hier_violation_rate_12":  f"{hier_violation_rate:.4f}",
            "test_label_cooc_consistency_12": f"{label_cooc_consistency:.4f}",
            "test_jaccard_macro_12":        f"{jaccard_score(y_true_12, y_pred_12, average='macro', zero_division=0):.4f}",
            "test_jaccard_micro_12":        f"{jaccard_score(y_true_12, y_pred_12, average='micro', zero_division=0):.4f}",
            "test_hamming_loss_12":         f"{hamming_loss(y_true_12, y_pred_12):.4f}",
            # misc
            "test_loss":                    f"{avg_loss:.4f}",
        })
        writer.writerow(out_row)
        out_file.flush()

        w = 62
        print(f"\n  {'─'*w}")
        print(f"  HMCN Fine — 138 labels  ({len(valid_cols_138)}/138 used for AUC)")
        print(f"  {'─'*w}")
        print(f"  {'ROC AUC':<32}: {auroc_138:.4f}")
        print(f"  {'PR AUC':<32}: {aucpr_138:.4f}")
        print(f"  {'F1 (macro)':<32}: {out_row['test_f1_macro_138']}")
        print(f"  {'F1 (micro)':<32}: {out_row['test_f1_micro_138']}")
        print(f"  {'Hier Violation Rate':<32}: {hier_violation_rate:.4f}")
        print(f"  {'Precision (micro)':<32}: {out_row['test_precision_micro_138']}")
        print(f"  {'Recall (micro)':<32}: {out_row['test_recall_micro_138']}")
        print(f"  {'Precision (macro)':<32}: {out_row['test_precision_macro_138']}")
        print(f"  {'Recall (macro)':<32}: {out_row['test_recall_macro_138']}")
        print(f"  {'Accuracy':<32}: {out_row['test_accuracy_138']}")
        print(f"  {'Jaccard (macro)':<32}: {out_row['test_jaccard_macro_138']}")
        print(f"  {'Jaccard (micro)':<32}: {out_row['test_jaccard_micro_138']}")
        print(f"  {'Hamming Loss':<32}: {out_row['test_hamming_loss_138']}")
        print(f"  {'─'*w}")
        print(f"  HMCN Meta — 12 groups   ({len(valid_cols_12)}/12 used for AUC)")
        print(f"  {'─'*w}")
        print(f"  {'ROC AUC':<32}: {auroc_12:.4f}")
        print(f"  {'PR AUC':<32}: {aucpr_12:.4f}")
        print(f"  {'F1 (macro)':<32}: {out_row['test_f1_macro_12']}")
        print(f"  {'F1 (micro)':<32}: {out_row['test_f1_micro_12']}")
        print(f"  {'Instance-F1':<32}: {out_row['test_f1_instance_12']}")
        print(f"  {'Balanced Accuracy':<32}: {bal_acc_12:.4f}")
        print(f"  {'Accuracy':<32}: {out_row['test_accuracy_12']}")
        print(f"  {'Sensitivity (macro)':<32}: {sensitivity:.4f}")
        print(f"  {'Specificity (macro)':<32}: {specificity:.4f}")
        print(f"  {'Hier Violation Rate':<32}: {hier_violation_rate:.4f}")
        print(f"  {'Label Co-occ Consistency':<32}: {label_cooc_consistency:.4f}")
        print(f"  {'Jaccard (macro)':<32}: {out_row['test_jaccard_macro_12']}")
        print(f"  {'Jaccard (micro)':<32}: {out_row['test_jaccard_micro_12']}")
        print(f"  {'Hamming Loss':<32}: {out_row['test_hamming_loss_12']}")
        print(f"  {'─'*w}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  {'─'*w}\n")

    out_file.close()
    print(f"\n{'='*60}\nDone. Test results saved to '{output_csv}'.")


# =============================================================================
# K-FOLD CROSS-VALIDATION
#
# Motivation: with ~5k molecules, a single fixed train/val/test split can give
# a metric that's more a reflection of "which molecules happened to land in
# test" than of the model config itself. Rather than repeating one frozen
# split many times (which, given `set_seed`, just reruns the same experiment),
# this rotates k *different* stratified splits, trains/evaluates the sweep on
# each, and reports mean ± std per config — an honest estimate of how much a
# given (lambda, heads, T_0, epochs) config's score actually moves depending
# on the split.
# =============================================================================

def build_graphs(df_split, label_start_col=2):
    """
    Convert a dataframe slice (with a 'nonStereoSMILES' column and label columns
    starting at `label_start_col`) into a list of PyG graphs, mirroring the
    module-level graph-building loops above. Skips unparseable SMILES.
    """
    graphs = []
    n_skipped = 0
    for i in range(len(df_split)):
        smiles = df_split['nonStereoSMILES'].iloc[i]
        y = torch.tensor(df_split.iloc[i, label_start_col:].to_numpy(dtype=float), dtype=torch.float)
        data = smiles_to_graph(smiles, y)
        if data is None:
            n_skipped += 1
            continue
        graphs.append(data)
    if n_skipped:
        print(f"    (skipped {n_skipped} unparseable SMILES in this split)")
    return graphs


def run_kfold_sweep(
    df,
    k                       = 5,
    val_ratio_within_train  = 0.15,   # carved out of each fold's (train+val) portion for model selection
    seed                    = SEED,
    # ── Sweep grid (forwarded to run_sweep for every fold) ─────────────────────
    lambda_values           = [0.0, 0.01, 0.05, 0.1],
    heads_values            = [1, 4, 8],
    t0_values               = [50, 100, 200],
    epochs_values           = [500, 1000],
    beta                    = 0.5,
    dropout                 = 0.47,
    learning_rate           = 1e-3,
    batch_size              = 128,
    num_classes             = 138,
    num_parents             = 12,
    monitor                 = "val_auroc",
    # ── Output ───────────────────────────────────────────────────────────────
    results_dir             = "kfold_results",
):
    """
    Runs the full run_sweep() + evaluate_sweep() pipeline across k stratified
    folds and aggregates results per config (mean ± std across folds).

    Fold splits are cached to disk (results_dir/kfold_split_indices.pkl), so
    rerunning this function reuses the exact same k partitions rather than
    generating new ones — the split is fixed per fold, only the sweep grid
    varies, keeping fold-to-fold comparisons apples-to-apples across separate
    script executions.
    """
    os.makedirs(results_dir, exist_ok=True)
    label_columns = df.iloc[:, 2:].values

    # ── Build (or load cached) k-fold train/val/test index partitions ─────────
    fold_cache_path = os.path.join(results_dir, "kfold_split_indices.pkl")
    if os.path.exists(fold_cache_path):
        with open(fold_cache_path, "rb") as f:
            fold_splits = pickle.load(f)
        print(f"Loaded cached {len(fold_splits)}-fold split from '{fold_cache_path}'.")
    else:
        X = np.arange(len(df)).reshape(-1, 1)
        np.random.seed(seed)
        outer = IterativeStratification(n_splits=k, order=2)
        outer_folds = list(outer.split(X, label_columns))

        fold_splits = []
        for fold_i, (trainval_idx, test_idx) in enumerate(outer_folds):
            np.random.seed(seed + fold_i + 1)   # distinct-but-deterministic per fold
            X_tv = np.arange(len(trainval_idx)).reshape(-1, 1)
            y_tv = label_columns[trainval_idx]
            inner = IterativeStratification(
                n_splits=2, order=2,
                sample_distribution_per_fold=[val_ratio_within_train, 1 - val_ratio_within_train],
            )
            train_rel, val_rel = next(inner.split(X_tv, y_tv))
            train_idx = trainval_idx[train_rel]
            val_idx   = trainval_idx[val_rel]
            fold_splits.append((train_idx, val_idx, test_idx))

        with open(fold_cache_path, "wb") as f:
            pickle.dump(fold_splits, f)
        print(f"Computed a fresh {k}-fold split and cached it to '{fold_cache_path}'.")

    # ── Run the sweep on each fold ──────────────────────────────────────────────
    per_fold_test_dfs = []

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(fold_splits):
        print(f"\n{'='*70}\nFOLD {fold_i+1}/{len(fold_splits)}  "
              f"({len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test)\n{'='*70}")

        df_train = df.iloc[train_idx].copy()
        df_val   = df.iloc[val_idx].copy()
        df_test  = df.iloc[test_idx].copy()

        print("  Building graphs...")
        fold_train_graphs = build_graphs(df_train)
        fold_val_graphs   = build_graphs(df_val)
        fold_test_graphs  = build_graphs(df_test)

        fold_results_csv = os.path.join(results_dir, f"fold{fold_i}_sweep_results.csv")
        fold_ckpt_dir    = os.path.join(results_dir, f"fold{fold_i}_checkpoints")
        fold_test_csv    = os.path.join(results_dir, f"fold{fold_i}_test_results.csv")

        run_sweep(
            train_data          = fold_train_graphs,
            val_data             = fold_val_graphs,
            df_graph_train       = fold_train_graphs,
            child_parent_pairs   = child_parent_pairs,
            child_idxs           = child_idxs,
            parent_idxs          = parent_idxs,
            num_node_features    = fold_train_graphs[0].x.shape[1],
            num_edge_features    = fold_train_graphs[0].edge_attr.shape[1],
            lambda_values        = lambda_values,
            heads_values         = heads_values,
            t0_values             = t0_values,
            epochs_values        = epochs_values,
            beta                  = beta,
            dropout               = dropout,
            learning_rate        = learning_rate,
            batch_size            = batch_size,
            num_classes           = num_classes,
            num_parents           = num_parents,
            results_csv           = fold_results_csv,
            checkpoint_dir        = fold_ckpt_dir,
            monitor               = monitor,
        )

        evaluate_sweep(
            sweep_csv            = fold_results_csv,
            checkpoint_dir        = fold_ckpt_dir,
            val_data              = fold_val_graphs,
            test_data             = fold_test_graphs,
            child_parent_pairs    = child_parent_pairs,
            num_node_features     = fold_train_graphs[0].x.shape[1],
            num_edge_features     = fold_train_graphs[0].edge_attr.shape[1],
            num_classes           = num_classes,
            num_parents           = num_parents,
            output_csv            = fold_test_csv,
            batch_size            = batch_size,
        )

        fold_df = pd.read_csv(fold_test_csv)
        fold_df.insert(0, "fold", fold_i)
        per_fold_test_dfs.append(fold_df)

    # ── Aggregate across folds ──────────────────────────────────────────────────
    combined = pd.concat(per_fold_test_dfs, ignore_index=True)
    combined_path = os.path.join(results_dir, "all_folds_test_results.csv")
    combined.to_csv(combined_path, index=False)

    config_cols = ["lambda_hier", "num_heads", "T_0", "epochs"]
    non_metric_cols = set(config_cols) | {"fold", "run_id", "timestamp", "beta", "dropout", "lr", "best_epoch", "monitor_metric"}
    metric_cols = [c for c in combined.columns if c not in non_metric_cols]

    summary = combined.groupby(config_cols)[metric_cols].agg(["mean", "std"])
    summary_path = os.path.join(results_dir, "kfold_summary_mean_std.csv")
    summary.to_csv(summary_path)

    print(f"\n{'='*70}\nK-fold sweep complete.")
    print(f"  Per-fold results:  '{combined_path}'")
    print(f"  Mean ± std summary: '{summary_path}'")
    print(f"{'='*70}")

    return combined, summary


# --- Single fixed-split run (kept for reference / quick debugging) -------------
# Uncomment to go back to a single train/val/test split instead of k-fold CV.
#
# run_sweep(
#     train_data         = train_data,
#     val_data           = val_data,
#     df_graph_train     = df_graph_train,
#     child_parent_pairs = child_parent_pairs,
#     child_idxs         = child_idxs,
#     parent_idxs        = parent_idxs,
#     num_node_features  = df_graph_train[0].x.shape[1],
#     num_edge_features  = df_graph_train[0].edge_attr.shape[1],
#     lambda_values      = [0.01],
#     heads_values       = [8],
#     t0_values          = [50],
#     epochs_values      = [100],
#     monitor            = "val_auroc",
#     results_csv        = "sweep_results_positionalEncoding_GATV2_HMCNF.csv",
#     checkpoint_dir     = "checkpoints",
# )
#
# evaluate_sweep(
#     sweep_csv          = "sweep_results_positionalEncoding_GATV2_HMCNF.csv",
#     checkpoint_dir     = "checkpoints",
#     val_data           = val_data,
#     test_data          = test_data,
#     child_parent_pairs = child_parent_pairs,
#     num_node_features  = df_graph_train[0].x.shape[1],
#     num_edge_features  = df_graph_train[0].edge_attr.shape[1],
#     output_csv         = "sweep_test_results_positionalEncoding_GATV2_HMCNF.csv",
# )

# --- K-fold cross-validated sweep (current) -------------------------------------
run_kfold_sweep(
    df                 = df,
    k                   = 5,
    val_ratio_within_train = 0.15,
    lambda_values      = [0, 0.01, 0.05, 0.1, 0.3],
    heads_values       = [8],
    t0_values          = [50],
    epochs_values      = [100],
    monitor            = "val_auroc",
    results_dir        = "kfold_results",
)

# --- Runtime tracking -------------------------------------------------------------
script_end_time = datetime.now()
elapsed = script_end_time - SCRIPT_START_TIME
print(f"\nScript finished at: {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total elapsed time: {elapsed}")
# -----------------------------------------------------------------------------------