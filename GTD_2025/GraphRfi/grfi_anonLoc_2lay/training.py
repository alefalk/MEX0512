import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from ndf import GTD100FeatureLayer, GTD200FeatureLayer, GTD300FeatureLayer, GTD478FeatureLayer, Forest, NeuralDecisionForest
from torchmetrics.classification import Precision, Recall, F1Score, AUROC
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import time
from haversine import haversine, Unit
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
from build_graph_data import *



class GCNClassifier(torch.nn.Module):
    def __init__(self, in_channels, h1, h2, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, h1)
        self.conv2 = GCNConv(h1, h2)
        self.conv3 = GCNConv(h2, num_classes)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Return embeddings (after first conv layer)
        embeddings = x

        # Second convolution layer (logits for classification)
        x = self.conv3(x, edge_index)
        
        return x, embeddings  # Return both logits (for classification) and embeddings

def split_metrics(model, x, edge_index, y, mask, num_classes, split_name):
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
    y_true = y[mask].detach().cpu().numpy()
    logits_np = logits[mask].detach().cpu().numpy()
    mets = compute_all_metrics(y_true, logits_np, num_classes)
    return {f"{split_name}_{k}": v for k, v in mets.items()}

    # OVR ROC-AUCs (robust if some classes missing in split)
    # If only one class present in y_true, roc_auc_score will raise; guard it.
    def safe_roc(avg):
        try:
            return roc_auc_score(y_true_np, proba_np, multi_class='ovr', average=avg, labels=labels)
        except Exception:
            return float('nan')

    out['roc_w']  = safe_roc('weighted')
    out['roc_mi'] = safe_roc('micro')
    out['roc_ma'] = safe_roc('macro')
    return out

def induced_subgraph(edge_index, keep_mask):
    keep = keep_mask.nonzero(as_tuple=False).view(-1)
    # map old -> new ids
    new_id = -torch.ones(keep_mask.size(0), dtype=torch.long, device=edge_index.device)
    new_id[keep] = torch.arange(keep.numel(), device=edge_index.device)
    src, dst = edge_index
    m = keep_mask[src] & keep_mask[dst]
    ei = edge_index[:, m]
    ei = torch.stack([new_id[ei[0]], new_id[ei[1]]], dim=0)
    return ei, new_id

def encode_labels_series(y_series, train_idx):
    """Fit encoder on TRAIN labels only; return encoded vector and mapping."""
    le = LabelEncoder()
    le.fit(y_series.loc[train_idx])
    y_all = pd.Series(index=y_series.index, data=-1, dtype=int)
    seen_mask = y_series.isin(le.classes_)
    y_all.loc[seen_mask] = le.transform(y_series.loc[seen_mask])
    label_index = {lab:i for i, lab in enumerate(le.classes_)}  # str -> int
    return y_all.astype(int), label_index, le

def to_device_tensor(x_np, device):
    return torch.tensor(x_np, dtype=torch.float32, device=device)


def get_metrics(val_true, val_pred, val_probs, val_acc, epoch):
    # Metrics on validation split
    prec_w = precision_score(val_true, val_pred, average="weighted", zero_division=0)
    rec_w  = recall_score(   val_true, val_pred, average="weighted", zero_division=0)
    f1_w   = f1_score(       val_true, val_pred, average="weighted")

    prec_mi = precision_score(val_true, val_pred, average="micro", zero_division=0)
    rec_mi  = recall_score(   val_true, val_pred, average="micro",   zero_division=0)
    f1_mi   = f1_score(       val_true, val_pred, average="micro")

    prec_ma = precision_score(val_true, val_pred, average="macro", zero_division=0)
    rec_ma  = recall_score(   val_true, val_pred, average="macro",  zero_division=0)
    f1_ma   = f1_score(       val_true, val_pred, average="macro")

    roc_auc_weighted = roc_auc_score(val_true, val_probs, multi_class='ovr', average='weighted')
    roc_auc_micro    = roc_auc_score(val_true, val_probs, multi_class='ovr', average='micro')
    roc_auc_macro    = roc_auc_score(val_true, val_probs, multi_class='ovr', average='macro')

    best_metrics = dict(
        acc=val_acc, epoch=epoch,
        prec_w=prec_w, rec_w=rec_w, f1_w=f1_w,
        prec_mi=prec_mi, rec_mi=rec_mi, f1_mi=f1_mi,
        prec_ma=prec_ma, rec_ma=rec_ma, f1_ma=f1_ma,
        roc_w=roc_auc_weighted, roc_mi=roc_auc_micro, roc_ma=roc_auc_macro,
    )

    return best_metrics


def train_joint_gcn_nrf(
    df_all,
    node_feature_cols,
    edge_feature_cols,
    edge_mode,
    equal_cols,
    k,
    nrf_cfg,                      # dict with forest hyperparams + partition name
    weight_decay=5e-4,
    device="cuda"
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split data
    train_idx, val_idx, test_idx = temporal_row_split(df_all, label_column='gname', train_ratio=0.6, val_ratio=0.2)

    ##################################### PREPARE GCN ############################################
    
    # Compute scalers
    node_scaler, edge_scaler = fit_scalers(
                df_all,
                node_feature_cols=node_feature_cols,
                edge_feature_cols=edge_feature_cols,
                train_idx=train_idx
    )

    # Build train + val graph
    trainval_idx = np.concatenate([train_idx, val_idx])
    x_tv, ei_tv, y_tv, _, _, _, label_index, chosen_radius = build_graph_on_index_set(
        df_all=df_all,
        idx_subset=trainval_idx,
        node_feature_cols=node_feature_cols,
        edge_feature_cols=edge_feature_cols,
        node_scaler=node_scaler, edge_scaler=edge_scaler,
        label_column='gname',
        train_idx=train_idx, val_idx=val_idx, test_idx=None,
        edge_mode=edge_mode, equal_cols=equal_cols, k=k, radius=None  # radius will be auto-picked from TRAIN in this set
    )

    # masks relative to the TRAIN+VAL subgraph:
    # (build them by mapping global indices -> local positions)
    pos_tv = {rid:i for i, rid in enumerate(trainval_idx)}
    train_mask_sub = torch.zeros(x_tv.size(0), dtype=torch.bool, device='cuda')
    val_mask_sub   = torch.zeros_like(train_mask_sub)
    for rid in train_idx: train_mask_sub[pos_tv[rid]] = True
    for rid in val_idx:   val_mask_sub[pos_tv[rid]]   = True

    y_all_encoded, label_index, le = encode_labels_series(df_all['gname'], train_idx)
    num_classes = len(label_index)

    ################################## Define models ##############################################

    gcn = GCNClassifier(in_channels=x_tv.size(1),
                       h1=nrf_cfg['h1'], h2=nrf_cfg['h2'],
                       num_classes=num_classes,
                       dropout=nrf_cfg['dropout']).to(device)

    # Select feature extractor
    input_size = len(df_all.columns) - 1 + nrf_cfg['h2']
    feat_layer_cls = {
        "gtd100": GTD100FeatureLayer,
        "gtd200": GTD200FeatureLayer,
        "gtd300": GTD300FeatureLayer,
        "gtd478": GTD478FeatureLayer,
    }[nrf_cfg['partition']]

    feat_layer = feat_layer_cls(out_layer_size = nrf_cfg['out_size_nrf'], input_size = input_size, dropout_rate=nrf_cfg['feat_dropout'])

    # Create neural decision forest
    forest = Forest(
        n_tree=nrf_cfg['n_tree'],
        tree_depth=nrf_cfg['tree_depth'],
        n_in_feature=feat_layer.get_out_feature_size(),
        tree_feature_rate=nrf_cfg['tree_feature_rate'],
        n_class=nrf_cfg['n_class']
    )
    neural_forest = NeuralDecisionForest(feat_layer, forest).to(device)

    y_tv_vec = torch.full((x_tv.size(0),), -100, dtype=torch.long, device=device)
    for rid in trainval_idx:
        enc = int(y_all_encoded.loc[rid])
        if enc >= 0:
            y_tv_vec[pos_tv[rid]] = enc

    X_tv_np = node_scaler.transform(df_all.loc[trainval_idx, node_feature_cols].fillna(0.0).to_numpy())
    X_tv_tab = to_device_tensor(X_tv_np, device)

    def nrf_loss_fn(prob, target):
        prob = prob.clamp(min=1e-8)
        return F.nll_loss(torch.log(prob), target)

    optimizer = torch.optim.AdamW(
        list(gcn.parameters()) + list(neural_forest.parameters()), lr=nrf_cfg['lr']
    )

    best_acc = -1
    best_epoch = -1
    best_state_dict = None
    epoch_logs = []
    no_improvement = 0

    patience = 300 if nrf_cfg.get('final_evaluation', True) else 100

    for epoch in range(nrf_cfg['epochs']):
        start_time = time.time()

        # Set train mode
        gcn.train(); neural_forest.train(); optimizer.zero_grad()

        # Forward pass GCN on subgraph
        logits_tv, emb_tv = gcn(x_tv, ei_tv)

        # Build input to NRF
        X_nrf_tv = torch.cat([X_tv_tab, emb_tv], dim=1)

        # Forward pass NRF
        prob_tv = neural_forest(X_nrf_tv)

        # compute joint loss
        loss_gcn = F.cross_entropy(logits_tv[train_mask_sub], y_tv_vec[train_mask_sub])
        y_train_targets = y_tv_vec[train_mask_sub]
        loss_nrf = nrf_loss_fn(prob_tv[train_mask_sub], y_train_targets)
        loss = loss_gcn + loss_nrf

        loss.backward()
        optimizer.step()

        # ---- Validation ----
        gcn.eval(); neural_forest.eval()
        with torch.no_grad():
            logits_tv_eval, emb_tv_eval = gcn(x_tv, ei_tv)
            X_nrf_tv_eval = torch.cat([X_tv_tab, emb_tv_eval], dim=1)
            prob_tv_eval = neural_forest(X_nrf_tv_eval)            # [M, C], probs (sum=1)

            # --- select VAL split in the subgraph order ---
            val_probs = prob_tv_eval[val_mask_sub].clamp_min(1e-8) # avoid log(0) elsewhere
            val_pred  = val_probs.argmax(dim=1)                    # [n_val]
            val_true  = y_tv_vec[val_mask_sub] 

            val_acc = (val_pred == val_true).float().mean().item()

            if val_acc > best_acc:
                best_acc   = val_acc
                best_epoch = epoch

                # Store CPU copies of weights
                best_state_dict = {
                    "gcn": {k: v.detach().cpu().clone() for k, v in gcn.state_dict().items()},
                    "ndf": {k: v.detach().cpu().clone() for k, v in neural_forest.state_dict().items()},
                }
                y_true_np  = val_true.detach().cpu().numpy()
                y_pred_np  = val_pred.detach().cpu().numpy()
                y_proba_np = val_probs.detach().cpu().numpy()

                best_metrics = get_metrics(y_true_np, y_pred_np, y_proba_np, val_acc, epoch)

                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

       
        epoch_logs.append(time.time() - start_time)
    
    if nrf_cfg.get("final_evaluation", True):
        # 1) Build FULL graph with SAME scalers and SAME edge params
        full_ids = df_all.index.to_numpy()
        x_full, ei_full, y_full, train_mask_full, val_mask_full, test_mask_full, _, _ = build_graph_on_index_set(
            df_all=df_all,
            idx_subset=full_ids,
            node_feature_cols=node_feature_cols,
            edge_feature_cols=edge_feature_cols,
            node_scaler=node_scaler, edge_scaler=edge_scaler,   # <- fitted on TRAIN
            label_column="gname",
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            edge_mode=edge_mode, equal_cols=equal_cols, k=k, radius=chosen_radius
        )

        # 2) Strict inference edges: keep TV→TV and TV→Test, drop anything starting from Test
        tv_mask_full = (train_mask_full | val_mask_full)
        ei_infer = make_infer_edge_index(ei_full, tv_mask_full, test_mask_full)
        from torch_geometric.utils import add_self_loops
        ei_infer, _ = add_self_loops(ei_infer, num_nodes=x_full.size(0))

        # 3) Load the best weights found during training
        if best_state_dict is not None:
            gcn.load_state_dict({k: v.to(device) for k, v in best_state_dict["gcn"].items()})
            neural_forest.load_state_dict({k: v.to(device) for k, v in best_state_dict["ndf"].items()})

        # 4) One forward pass: GCN -> embeddings; concat with tabular; NRF -> proba
        gcn.eval(); neural_forest.eval()
        with torch.no_grad():
            # GCN forward on full graph with strict edges
            logits_full, emb_full = gcn(x_full, ei_infer)   # logits_full not used for final score; emb_full is

            # Tabular features for NRF must be scaled with the SAME node scaler and same column order
            X_full_tab_np = node_scaler.transform(
                df_all.loc[full_ids, node_feature_cols].fillna(0.0).to_numpy()
            )
            X_full_tab = torch.tensor(X_full_tab_np, dtype=torch.float32, device=device)

            # Concatenate [tabular || GCN embeddings] and predict with NRF
            X_nrf_full = torch.cat([X_full_tab, emb_full], dim=1)
            prob_full = neural_forest(X_nrf_full)  # [N, C], probabilities that sum to 1

            # 5) Metrics on TEST only
            test_mask = test_mask_full
            y_test_true = y_full[test_mask].detach().cpu().numpy()              # encoded with train label space; -100 ignored below if needed
            y_test_pred = prob_full[test_mask].argmax(dim=1).detach().cpu().numpy()
            y_test_proba = prob_full[test_mask].detach().cpu().numpy()

            test_acc = accuracy_score(y_test_true, y_test_pred)

            best_metrics = get_metrics(y_test_true, y_test_pred, y_test_proba, test_acc, best_epoch)
            print(f"Best test acc: {test_acc:.4f} @ epoch {best_epoch}")
    else:
        print(f"Best validation acc: {best_acc:.4f} @ epoch {best_epoch}")
        test_acc = best_acc

    return test_acc, best_epoch, best_metrics, epoch_logs

