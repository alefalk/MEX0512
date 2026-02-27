# graph_utils_transductive.py
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Sequence
from haversine import haversine, Unit
from sklearn.neighbors import BallTree

import torch
from torch_geometric.utils import to_undirected, add_self_loops
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


EARTH_RADIUS_KM = 6371.0088  # WGS84 mean


# ----------------------------
# 1) Splitting without leakage
# ----------------------------

def temporal_row_split(
    df: pd.DataFrame,
    label_column: str = "gname",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified temporal split by label at the ROW (node) level.
    Keeps chronological order within each label group to avoid temporal leakage.
    """
    idx_train, idx_val, idx_test = [], [], []
    for _, group in df.groupby(label_column):
        n = len(group)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))
        idx_train.append(group.index[:train_end].to_numpy())
        idx_val.append(group.index[train_end:val_end].to_numpy())
        idx_test.append(group.index[val_end:].to_numpy())
    train_idx = np.concatenate(idx_train) if idx_train else np.array([], dtype=int)
    val_idx   = np.concatenate(idx_val)   if idx_val   else np.array([], dtype=int)
    test_idx  = np.concatenate(idx_test)  if idx_test  else np.array([], dtype=int)
    return train_idx, val_idx, test_idx



# ---------------------------------------
# 2) Label space + tensors without leak
# ---------------------------------------

def make_label_index_from_train(
    df: pd.DataFrame,
    train_idx: Sequence[int],
    label_column: str = "gname",
) -> dict:
    """
    Create label_index ONLY from train labels (no test leakage).
    """
    train_labels = df.loc[train_idx, label_column].dropna().unique().tolist()
    return {lab: i for i, lab in enumerate(sorted(train_labels))}


# ---------------------------------
# 3) Global graph (row = one node)
# ---------------------------------

def _equal_edges(df_all: pd.DataFrame, equal_cols, device="cuda", max_full_clique_size: int = 1000):
    """
    Build edges by connecting nodes that share identical values on equal_cols.
    Creates cliques per group. For very large groups, you may want to cap or sample.
    """
    import numpy as np, torch
    rows, cols = [], []
    for _, g in df_all.groupby(equal_cols, sort=False):
        idx = g.index.to_numpy()                    # row IDs (original df indices)
        n = len(idx)
        if n <= 1:
            continue
        # Map original row IDs to contiguous positions 0..N-1 in this batch:
        # We'll translate to global positions outside using a dict, but since
        # build_global_graph_transductive already has node_ids and row_pos, we can
        # directly use df indices as positions via that mapping later. For speed, we
        # create pair indices locally (0..n-1) and convert to df indices afterwards.

        # all unordered pairs i<j
        I, J = np.triu_indices(n, k=1)
        rows.extend(idx[I]); cols.extend(idx[J])
        rows.extend(idx[J]); cols.extend(idx[I])    # make it symmetric
    if not rows:
        return torch.empty((2,0), dtype=torch.long, device=device)
    # We'll remap these df indices to [0..N-1] later using row_pos
    return torch.tensor([rows, cols], dtype=torch.long, device=device)

def make_infer_edge_index_strict(edge_index, trainval_mask, test_mask):
    """
    Keep: Train/Val → Train/Val  and  Train/Val → Test.
    Drop everything starting from Test (so no Test→Train, no Test→Test).
    """
    src, dst = edge_index
    keep = (trainval_mask[src] & trainval_mask[dst]) | (trainval_mask[src] & test_mask[dst])
    return edge_index[:, keep]


# --- 0A) Fit scalers once (NO leakage: fit on TRAIN only) ---
def fit_scalers(df_all, node_feature_cols, edge_feature_cols, train_idx):
    node_feats = df_all[node_feature_cols].fillna(0.0).astype(np.float32)
    #edge_feats = df_all[edge_feature_cols].fillna(0.0).astype(np.float32)
    node_scaler = StandardScaler().fit(node_feats.loc[train_idx].to_numpy())
    #edge_scaler = StandardScaler().fit(edge_feats.loc[train_idx].to_numpy())
    return node_scaler

# --- 0B) Build graph on an arbitrary index set (e.g., train+val OR full) ---
def build_graph_on_index_set(
    df_all, idx_subset,                      # rows to include as nodes (e.g., train+val indices OR all indices)
    node_feature_cols, edge_feature_cols,
    node_scaler, edge_scaler,                # prefit on TRAIN only
    label_column,                            # "gname"
    train_idx=None, val_idx=None, test_idx=None,   # full-index arrays (optional; used to derive masks if subset==full)
    edge_mode="radius", equal_cols=None, k=10, radius=None
):
    """
    Returns: x, edge_index, y, masks (if full was passed), label_index, chosen_radius
    - If idx_subset == full df indices and you pass train/val/test idx, masks are constructed.
    - If idx_subset is train+val only, only train/val masks (rel to this subgraph) are returned; test mask is None.
    - chosen_radius: the epsilon actually used (so you can reuse it on the full graph).
    """
    device = "cuda"
    import numpy as np, torch
    from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
    from torch_geometric.utils import to_undirected, add_self_loops

    # map subset rows to 0..M-1
    sub_ids = np.array(idx_subset, dtype=int)
    pos = {rid: i for i, rid in enumerate(sub_ids)}
    M = len(sub_ids)

    # transform features using TRAIN-fitted scalers
    node_X = node_scaler.transform(df_all.loc[sub_ids, node_feature_cols].fillna(0.0).astype(np.float32).to_numpy())
    #edge_Z = edge_scaler.transform(df_all.loc[sub_ids, edge_feature_cols].fillna(0.0).astype(np.float32).to_numpy())
    x = torch.tensor(node_X, dtype=torch.float32, device=device)

    # label space from TRAIN (global, not subset)
    train_labels = df_all.loc[train_idx, label_column].dropna().unique().tolist() if train_idx is not None else []
    label_index = {lab: i for i, lab in enumerate(sorted(train_labels))}
    y = torch.full((M,), -100, dtype=torch.long, device=device)
    for rid in sub_ids:
        lab = df_all.at[rid, label_column]
        if lab in label_index:
            y[pos[rid]] = label_index[lab]

    # edge construction (same logic you already have, but restricted to sub_ids)
    edges = []
    chosen_radius = radius

    if edge_mode == "geo_equal_radius":
        if not equal_cols:
            raise ValueError("equal_cols required (e.g., ['longitude','latitude'])")

        df_sub = df_all.loc[sub_ids]
        lat_col = "latitude"
        lon_col = "longitude"
        if lat_col not in df_sub or lon_col not in df_sub:
            raise ValueError(f"Missing columns '{lat_col}'/'{lon_col}' in df_all.")

        # ---------- 1) EXACT equality edges ----------
        ei_eq_df = _equal_edges(df_sub, equal_cols=equal_cols, device=device)
        already = set()  # track undirected pairs to avoid duplicates
        if ei_eq_df.numel() > 0:
            r_eq = torch.tensor([pos[int(i)] for i in ei_eq_df[0].tolist()], device=device)
            c_eq = torch.tensor([pos[int(i)] for i in ei_eq_df[1].tolist()], device=device)
            ei_eq = torch.stack([r_eq, c_eq], dim=0)
            edges.append(ei_eq)
            for u, v in zip(r_eq.tolist(), c_eq.tolist()):
                a, b = (u, v) if u < v else (v, u)
                already.add((a, b))

        # ---------- 2) RADIUS (<= 1 km) edges ----------
        radius_km = 1.0
        use_balltree = True  # set False to use the simple O(M^2) loop

        rows, cols = [], []

        if use_balltree:
            # BallTree expects [lat, lon] in radians
            lat = df_sub[lat_col].to_numpy(dtype=float)
            lon = df_sub[lon_col].to_numpy(dtype=float)
            X_rad = np.c_[np.deg2rad(lat), np.deg2rad(lon)]
            tree = BallTree(X_rad, metric="haversine")
            radius_rad = radius_km / EARTH_RADIUS_KM

            neigh_ind = tree.query_radius(X_rad, r=radius_rad, return_distance=False)
            for i, nbrs in enumerate(neigh_ind):
                for j in nbrs:
                    if j == i:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    if (a, b) in already:
                        continue  # skip pairs already connected by equality
                    # add both directions
                    rows += [i, j]
                    cols += [j, i]
                    already.add((a, b))
        else:
            # Simple pairwise loop (O(M^2)). coords as (lat, lon)
            coords = list(zip(df_sub[lat_col].astype(float), df_sub[lon_col].astype(float)))
            M = len(coords)
            for i in range(M):
                for j in range(i+1, M):
                    a, b = (i, j)
                    if (a, b) in already:
                        continue  # equality already connected these
                    if haversine(coords[i], coords[j], unit=Unit.KILOMETERS) <= radius_km:
                        rows += [i, j]
                        cols += [j, i]
                        already.add((a, b))

        if rows:
            ei_rad = torch.tensor([rows, cols], dtype=torch.long, device=device)
            edges.append(ei_rad)


    edge_index = torch.cat(edges, dim=1) if edges else torch.empty((2,0), dtype=torch.long, device=device)
    edge_index = to_undirected(edge_index, num_nodes=M)
    edge_index, _ = add_self_loops(edge_index, num_nodes=M)

    """# optional: fix isolated (only self-loop)
    deg = torch.bincount(edge_index[0], minlength=M)
    iso = (deg == 1)
    if iso.any():
        nbrs1 = NearestNeighbors(n_neighbors=min(2, max(2, M))).fit(edge_Z)
        _, idx1 = nbrs1.kneighbors(edge_Z)
        iso_idx = torch.nonzero(iso, as_tuple=False).view(-1).cpu().numpy()
        rows_fix, cols_fix = [], []
        for i in iso_idx:
            j = int(idx1[i, 1])
            rows_fix += [i, j]; cols_fix += [j, i]
        fix_ei = torch.tensor([rows_fix, cols_fix], dtype=torch.long, device=device)
        edge_index = torch.cat([edge_index, fix_ei], dim=1)
        edge_index = to_undirected(edge_index, num_nodes=M)
        edge_index, _ = add_self_loops(edge_index, num_nodes=M)"""

    # masks (only if you passed full idxs && subset==full)
    train_mask_sub = val_mask_sub = test_mask_sub = None
    if (train_idx is not None) and (val_idx is not None) and (test_idx is not None):
        if len(idx_subset) == len(df_all):  # full graph case
            train_mask_sub = torch.zeros(M, dtype=torch.bool, device=device)
            val_mask_sub   = torch.zeros(M, dtype=torch.bool, device=device)
            test_mask_sub  = torch.zeros(M, dtype=torch.bool, device=device)
            for rid in train_idx: 
                if rid in pos: train_mask_sub[pos[rid]] = True
            for rid in val_idx: 
                if rid in pos: val_mask_sub[pos[rid]] = True
            for rid in test_idx: 
                if rid in pos: test_mask_sub[pos[rid]] = True

    return x, edge_index, y, train_mask_sub, val_mask_sub, test_mask_sub, label_index, chosen_radius
