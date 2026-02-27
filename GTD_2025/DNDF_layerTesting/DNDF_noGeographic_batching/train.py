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
from haversine import haversine, Unit
from collections import defaultdict, Counter
from torch.utils.data import DataLoader, TensorDataset



def build_graph_data(df, label_index):
    # Split rows to avoid leakage (your function)
    train_df, val_df, test_df = handle_leakage(df)

    # Row-level labels for NRF
    y_nrf = torch.tensor(df['gname'].map(label_index).values, dtype=torch.long)

    # NRF input features: everything except labels and the pair-defining cols
    drop_cols = {'gname'}
    non_label_cols = [c for c in df.columns if c not in drop_cols]
    nrf_input = torch.tensor(df[non_label_cols].astype(float).values, dtype=torch.float32)

    index_to_label = {v: k for k, v in label_index.items()}


    # Return structure kept compatible
    return (y_nrf, 
            train_df,
            val_df,
            test_df,
            index_to_label)


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

def train_joint(df, args, label_index, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Stratified-by-group split (your helper)
    train_df, val_df, test_df = handle_leakage(df)

    # Labels (full df row order)
    y_nrf = torch.tensor(df['gname'].map(label_index).values, dtype=torch.long, device=device)
    index_to_label = {v: k for k, v in label_index.items()}

    # ---- Feature extractor selection ----
    input_size = len(df.columns) - 1  # everything except 'gname'
    feat_layer_cls = {
        "gtd100": GTD100FeatureLayer,
        "gtd200": GTD200FeatureLayer,
        "gtd300": GTD300FeatureLayer,
        "gtd478": GTD478FeatureLayer,
    }[args['partition']]

    feat_layer = feat_layer_cls(
        out_layer_size=args['out_size_nrf'],
        input_size=input_size,
        dropout_rate=args['feat_dropout']
    )

    forest = Forest(
        n_tree=args['n_tree'],
        tree_depth=args['tree_depth'],
        n_in_feature=feat_layer.get_out_feature_size(),
        tree_feature_rate=args['tree_feature_rate'],
        n_class=args['n_class']
    )
    neural_forest = NeuralDecisionForest(feat_layer, forest).to(device)
    optimizer = torch.optim.AdamW(neural_forest.parameters(), lr=args['lr'])

    # ---- Build the full feature tensor once (same row order as df) ----
    non_location_cols = [c for c in df.columns if c != "gname"]
    non_geo_features = torch.tensor(
        df[non_location_cols].astype(float).values,
        dtype=torch.float32,
        device=device
    )

    # ---- Index tensors (CPU is fine for DataLoader samples; we move to device when selecting) ----
    train_idx_all = torch.tensor(train_df.index.values, dtype=torch.long)
    val_idx_all   = torch.tensor(val_df.index.values,   dtype=torch.long)
    test_idx_all  = torch.tensor(test_df.index.values,  dtype=torch.long)

    # ---- DataLoaders over row indices ----
    bs = args['batch_size']
    train_loader = DataLoader(TensorDataset(train_idx_all), batch_size=bs, shuffle=True, drop_last=False)
    val_loader   = DataLoader(TensorDataset(val_idx_all),   batch_size=bs, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TensorDataset(test_idx_all),  batch_size=bs, shuffle=False, drop_last=False)

    # ---- Training loop with early stopping on val acc ----
    best_acc = -1.0
    best_epoch = -1
    best_state_dict = None
    best_precision = best_recall = best_f1 = float('nan')
    best_precision_micro = best_recall_micro = best_f1_micro = float('nan')
    best_precision_macro = best_recall_macro = best_f1_macro = float('nan')
    roc_auc_weighted = roc_auc_micro = roc_auc_macro = float('nan')
    epoch_logs = []
    no_improvement = 0
    patience = 300 if args.get('final_evaluation', True) else 100

    for epoch in range(args['epochs']):
        start_time = time.time()
        neural_forest.train()
        running_loss = 0.0

        for (batch_idx_cpu,) in train_loader:
            # Move row indices to device and slice batch features/labels
            batch_idx = batch_idx_cpu.to(device, non_blocking=True)
            feat_batch = non_geo_features.index_select(0, batch_idx)
            out_batch = neural_forest(feat_batch)

            y_batch = y_nrf.index_select(0, batch_idx)

            loss = neural_forest.loss(out_batch, y_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_idx.size(0)

        train_loss_epoch = running_loss / len(train_idx_all)

        # ---- Validation ----
        neural_forest.eval()
        with torch.no_grad():
            val_logits_list = []
            val_targets_list = []
            for (batch_idx_cpu,) in val_loader:
                batch_idx = batch_idx_cpu.to(device, non_blocking=True)
                feat_batch = non_geo_features.index_select(0, batch_idx)
                out_val_batch = neural_forest(feat_batch)
                val_logits_list.append(out_val_batch)
                val_targets_list.append(y_nrf.index_select(0, batch_idx))

            out_val = torch.cat(val_logits_list, dim=0)
            y_val_true = torch.cat(val_targets_list, dim=0)

            pred_labels = out_val.argmax(dim=1)
            acc = (pred_labels == y_val_true).float().mean().item()

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_state_dict = {'ndf': neural_forest.state_dict()}

                # Metrics on validation split
                # Use torchmetrics on device for preds/targets
                best_precision = Precision(task='multiclass', average='weighted', num_classes=args['n_class']).to(device)(pred_labels, y_val_true).item()
                best_recall    = Recall(   task='multiclass', average='weighted', num_classes=args['n_class']).to(device)(pred_labels, y_val_true).item()
                best_f1        = F1Score(  task='multiclass', average='weighted', num_classes=args['n_class']).to(device)(pred_labels, y_val_true).item()

                best_precision_micro = Precision(task='multiclass', average='micro', num_classes=args['n_class']).to(device)(pred_labels, y_val_true).item()
                best_recall_micro    = Recall(   task='multiclass', average='micro', num_classes=args['n_class']).to(device)(pred_labels, y_val_true).item()
                best_f1_micro        = F1Score(  task='multiclass', average='micro', num_classes=args['n_class']).to(device)(pred_labels, y_val_true).item()

                best_precision_macro = Precision(task='multiclass', average='macro', num_classes=args['n_class']).to(device)(pred_labels, y_val_true).item()
                best_recall_macro    = Recall(   task='multiclass', average='macro', num_classes=args['n_class']).to(device)(pred_labels, y_val_true).item()
                best_f1_macro        = F1Score(  task='multiclass', average='macro', num_classes=args['n_class']).to(device)(pred_labels, y_val_true).item()

                # ROC-AUC (OVR) on validation
                y_proba_val = F.softmax(out_val, dim=1).detach().cpu().numpy()
                y_true_val  = y_val_true.detach().cpu().numpy()
                try:
                    roc_auc_weighted = roc_auc_score(y_true_val, y_proba_val, multi_class='ovr', average='weighted')
                    roc_auc_micro    = roc_auc_score(y_true_val, y_proba_val, multi_class='ovr', average='micro')
                    roc_auc_macro    = roc_auc_score(y_true_val, y_proba_val, multi_class='ovr', average='macro')
                except ValueError:
                    # Handle rare case where a class is missing in the current val split
                    roc_auc_weighted = roc_auc_micro = roc_auc_macro = float('nan')

                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        if verbose and (epoch % 50 == 0 or epoch == args['epochs'] - 1):
            print(f"Epoch {epoch:03d} | NRF Loss: {train_loss_epoch:.4f} | Val Acc: {acc:.4f}")

        epoch_logs.append(time.time() - start_time)

    if verbose:
        print(f"Best validation acc: {best_acc:.4f} @ epoch {best_epoch}")

    # -------------------- Final evaluation on test set --------------------
    if args.get('final_evaluation', True) and best_state_dict is not None:
        if verbose:
            print("Evaluating on test set using full graph...")
        neural_forest.load_state_dict(best_state_dict['ndf'])
        neural_forest.eval()

        with torch.no_grad():
            test_logits_list = []
            test_targets_list = []
            for (batch_idx_cpu,) in test_loader:
                batch_idx = batch_idx_cpu.to(device, non_blocking=True)
                feat_batch = non_geo_features.index_select(0, batch_idx)
                out_test_batch = neural_forest(feat_batch)
                test_logits_list.append(out_test_batch)
                test_targets_list.append(y_nrf.index_select(0, batch_idx))

            out_test = torch.cat(test_logits_list, dim=0)
            y_test_true = torch.cat(test_targets_list, dim=0)

            pred_labels_test = out_test.argmax(dim=1)
            test_acc = (pred_labels_test == y_test_true).float().mean().item()

            y_pred = pred_labels_test.detach().cpu().numpy()
            y_true = y_test_true.detach().cpu().numpy()

            y_pred_decoded = [index_to_label[i] for i in y_pred]
            y_true_decoded = [index_to_label[i] for i in y_true]
    else:
        test_acc = best_acc
        y_pred_decoded = []
        y_true_decoded = []

    return (
        test_acc, best_epoch, best_precision, best_recall, best_f1,
        y_pred_decoded, y_true_decoded,
        best_precision_micro, best_recall_micro, best_f1_micro,
        best_precision_macro, best_recall_macro, best_f1_macro,
        roc_auc_weighted, roc_auc_micro, roc_auc_macro,
        epoch_logs
    )
