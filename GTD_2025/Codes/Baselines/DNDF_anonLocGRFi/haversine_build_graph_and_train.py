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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import time

import torch
import pandas as pd
from sklearn.utils import shuffle

def build_nrf_data(df, label_index, feature_cols):
    # Split (by gname) exactly as before
    train_df, val_df, test_df = handle_leakage(df)

    # Build boolean masks on the ORIGINAL df index (no reset_index here)
    idx_train = set(train_df.index)
    idx_val   = set(val_df.index)
    idx_test  = set(test_df.index)

    n = len(df)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        if i in idx_train: train_mask[i] = True
        elif i in idx_val: val_mask[i] = True
        elif i in idx_test: test_mask[i] = True

    # Features + labels
    X = torch.tensor(df[feature_cols].astype(float).fillna(0).values, dtype=torch.float32)
    y_nrf = torch.tensor(df['gname'].map(label_index).values, dtype=torch.long)
    index_to_label = {v: k for k, v in label_index.items()}
    return X, y_nrf, train_mask, val_mask, test_mask, index_to_label


def handle_leakage(df):
    train_frames = []
    val_frames = []
    test_frames = []

    for _, group in df.groupby('gname'):

        n = len(group)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        train_frames.append(group.iloc[:train_end])
        val_frames.append(group.iloc[train_end:val_end])
        test_frames.append(group.iloc[val_end:])

    train_df = pd.concat(train_frames)
    val_df = pd.concat(val_frames)
    test_df = pd.concat(test_frames)

    return shuffle(train_df), shuffle(val_df), shuffle(test_df)

def train_joint(non_geo_features, y_nrf, train_mask, val_mask, test_mask, args, index_to_label, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Feature extractor (make sure input dim matches your feature_cols) ----
    feat_layer = {
        "gtd100": GTD100FeatureLayer,  # expects 47-d input
        "gtd200": GTD200FeatureLayer,  # expects 15-d input
        "gtd300": GTD300FeatureLayer,
        "gtd478": GTD478FeatureLayer,
    }[args['partition']](dropout_rate=args['feat_dropout'])

    forest = Forest(
        n_tree=args['n_tree'],
        tree_depth=args['tree_depth'],
        n_in_feature=feat_layer.get_out_feature_size(),  # 1024
        tree_feature_rate=args['tree_feature_rate'],
        n_class=args['n_class']
    )
    ndf = NeuralDecisionForest(feat_layer, forest).to(device)

    optimizer = torch.optim.AdamW(ndf.parameters(), lr=args['lr'])

    # Tensors to device
    X  = non_geo_features.to(device)
    y  = y_nrf.to(device)
    tr = train_mask.to(device)
    va = val_mask.to(device)
    te = test_mask.to(device)

    # Tracking
    patience = 300 if args.get('final_evaluation', True) else 100
    best_acc = -1.0
    best_epoch = -1
    best_state = None
    epoch_logs = []

    # These will be filled when best val improves
    best_precision = best_recall = best_f1 = 0.0
    best_precision_micro = best_recall_micro = best_f1_micro = 0.0
    best_precision_macro = best_recall_macro = best_f1_macro = 0.0
    roc_auc_weighted = roc_auc_micro = roc_auc_macro = 0.0

    no_improve = 0
    ncls = args['n_class']

    for epoch in range(args['epochs']):
        start_time = time.time()

        # ---- Train ----
        ndf.train()
        optimizer.zero_grad()
        out = ndf(X)                                # probs (N, C)
        loss = ndf.loss(out[tr], y[tr])             # NLLLoss(log(probs)) inside
        loss.backward()
        optimizer.step()

        # ---- Val ----
        ndf.eval()
        with torch.no_grad():
            out_v = ndf(X)                          # probs (N, C)
            pred_v = out_v[va].argmax(dim=1)
            acc_v  = (pred_v == y[va]).float().mean().item()

        # Early stopping bookkeeping
        if acc_v > best_acc:
            best_acc = acc_v
            best_epoch = epoch
            best_state = {'ndf': ndf.state_dict()}
            no_improve = 0

            # ---- Metrics on the *validation split* (using current best) ----
            with torch.no_grad():
                probs_va = out_v[va]                # (|val|, C)
                pred_labels = probs_va.argmax(dim=1)
                y_true_tensor = y[va]

                # Save numpy for AUC
                y_true_np = y_true_tensor.detach().cpu().numpy()
                y_proba_np = probs_va.detach().cpu().numpy()

                # Torchmetrics (weighted/micro/macro)
                prec_w = Precision(task='multiclass', average='weighted', num_classes=ncls).to(device)
                rec_w  = Recall(   task='multiclass', average='weighted', num_classes=ncls).to(device)
                f1_w   = F1Score(  task='multiclass', average='weighted', num_classes=ncls).to(device)

                prec_mi = Precision(task='multiclass', average='micro', num_classes=ncls).to(device)
                rec_mi  = Recall(   task='multiclass', average='micro', num_classes=ncls).to(device)
                f1_mi   = F1Score(  task='multiclass', average='micro', num_classes=ncls).to(device)

                prec_ma = Precision(task='multiclass', average='macro', num_classes=ncls).to(device)
                rec_ma  = Recall(   task='multiclass', average='macro', num_classes=ncls).to(device)
                f1_ma   = F1Score(  task='multiclass', average='macro', num_classes=ncls).to(device)

                best_precision      = prec_w(pred_labels, y_true_tensor).item()
                best_recall         = rec_w(pred_labels, y_true_tensor).item()
                best_f1             = f1_w(pred_labels, y_true_tensor).item()

                best_precision_micro = prec_mi(pred_labels, y_true_tensor).item()
                best_recall_micro    = rec_mi(pred_labels, y_true_tensor).item()
                best_f1_micro        = f1_mi(pred_labels, y_true_tensor).item()

                best_precision_macro = prec_ma(pred_labels, y_true_tensor).item()
                best_recall_macro    = rec_ma(pred_labels, y_true_tensor).item()
                best_f1_macro        = f1_ma(pred_labels, y_true_tensor).item()

                try:
                    roc_auc_weighted = roc_auc_score(y_true_np, y_proba_np, multi_class='ovr', average='weighted')
                    roc_auc_micro    = roc_auc_score(y_true_np, y_proba_np, multi_class='ovr', average='micro')
                    roc_auc_macro    = roc_auc_score(y_true_np, y_proba_np, multi_class='ovr', average='macro')
                except ValueError:
                    # AUC can fail if some classes are missing in val split; keep zeros
                    roc_auc_weighted = roc_auc_weighted
                    roc_auc_micro    = roc_auc_micro
                    roc_auc_macro    = roc_auc_macro
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | NDF Loss: {loss.item():.4f} | Val Acc: {acc_v:.4f}")

        epoch_logs.append(time.time() - start_time)

    print(f"Best validation acc: {best_acc:.4f} @ epoch {best_epoch}")

    # ---- Test or return best val ----
    if args.get('final_evaluation', True) and best_state is not None:
        ndf.load_state_dict(best_state['ndf'])
        ndf.eval()
        with torch.no_grad():
            out_t = ndf(X)
            probs_te = out_t[te]
            pred_t = probs_te.argmax(dim=1)
            test_acc = (pred_t == y[te]).float().mean().item()

            y_pred = pred_t.cpu().numpy()
            y_true = y[te].cpu().numpy()
            y_pred_decoded = [index_to_label[i] for i in y_pred]
            y_true_decoded = [index_to_label[i] for i in y_true]
    else:
        test_acc = best_acc
        # Return val decoded labels for consistency when not final
        with torch.no_grad():
            out_v = ndf(X)
            probs_va = out_v[va]
            pred_v = probs_va.argmax(dim=1)
            y_pred_decoded = [index_to_label[i.item()] for i in pred_v.cpu()]
            y_true_decoded = [index_to_label[i.item()] for i in y[va].cpu()]

    return (
        test_acc, best_epoch, best_precision, best_recall, best_f1,
        y_pred_decoded, y_true_decoded,
        best_precision_micro, best_recall_micro, best_f1_micro,
        best_precision_macro, best_recall_macro, best_f1_macro,
        roc_auc_weighted, roc_auc_micro, roc_auc_macro,
        epoch_logs
    )