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


    train_df, val_df, test_df = handle_leakage(df)
    y_nrf = torch.tensor(df['gname'].map(label_index).values, dtype=torch.long)

    index_to_label = {v: k for k, v in label_index.items()}

    # Select feature extractor
    input_size = len(df.columns) - 1
    feat_layer_cls = {
        "gtd100": GTD100FeatureLayer,
        "gtd200": GTD200FeatureLayer,
        "gtd300": GTD300FeatureLayer,
        "gtd478": GTD478FeatureLayer,
    }[args['partition']]

    feat_layer = feat_layer_cls(out_layer_size = args['out_size_nrf'], input_size = input_size, dropout_rate=args['feat_dropout'])

    # Create neural decision forest
    forest = Forest(
        n_tree=args['n_tree'],
        tree_depth=args['tree_depth'],
        n_in_feature=feat_layer.get_out_feature_size(),
        tree_feature_rate=args['tree_feature_rate'],
        n_class=args['n_class']
    )
    neural_forest = NeuralDecisionForest(feat_layer, forest).to(device)

    optimizer = torch.optim.AdamW(list(neural_forest.parameters()), lr=args['lr'])

    # Dtypes and device
    y_nrf = y_nrf.to(device).long()
    #non_geo_features = non_geo_features.to(device)

    best_acc = -1
    best_epoch = -1
    best_state_dict = None
    epoch_logs = []
    no_improvement = 0

    patience = 300 if args.get('final_evaluation', True) else 100

    for epoch in range(args['epochs']):
        start_time = time.time()

        neural_forest.train()
        optimizer.zero_grad()

        # assume you already decided which columns are non-geo features
        non_location_cols = [c for c in df.columns if c not in ["gname"]]

        # Build the full (all-rows) feature tensor once, in df row order
        non_geo_features = torch.tensor(
            df[non_location_cols].astype(float).values, dtype=torch.float32, device=device
        )

        # === Use train_df directly ===
        train_row_idx = torch.tensor(train_df.index.values, dtype=torch.long, device=device)

        feat_train = non_geo_features.index_select(0, train_row_idx)
        out_forest_train = neural_forest(feat_train)


        # Losses
        train_idx = torch.as_tensor(train_df.index.values, dtype=torch.long, device=device)
        loss2 = neural_forest.loss(out_forest_train, y_nrf.index_select(0, train_idx))
        #loss2 = neural_forest.loss(out_forest_train, y_nrf[row_train_mask])  # row-level NRF loss on train rows
        loss  = loss2

        loss.backward()
        optimizer.step()

        # ---- Validation ----
        neural_forest.eval()
        with torch.no_grad():
            val_row_idx = torch.tensor(val_df.index.values, dtype=torch.long, device=device)
            feat_val = non_geo_features.index_select(0, val_row_idx)
            
            out_val = neural_forest(feat_val)

            pred_labels = out_val.argmax(dim=1)
            
            val_idx = torch.as_tensor(val_df.index.values, dtype=torch.long, device=device)
            y_val_true = y_nrf.index_select(0, val_idx)
            acc = (pred_labels == y_val_true).float().mean().item()

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_labels = pred_labels
                best_proba = F.softmax(out_val, dim=1)
                best_state_dict = {
                    'ndf': neural_forest.state_dict()
                }
                no_improvement = 0

                # Metrics on validation split
                y_true_tensor = y_nrf.index_select(0, val_idx)
                y_pred_tensor = pred_labels
                y_proba = best_proba.detach().cpu().numpy()
                y_true = y_true_tensor.detach().cpu().numpy()

                best_precision = Precision(task='multiclass', average='weighted', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()
                best_recall    = Recall(   task='multiclass', average='weighted', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()
                best_f1        = F1Score(  task='multiclass', average='weighted', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()

                best_precision_micro = Precision(task='multiclass', average='micro', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()
                best_recall_micro    = Recall(   task='multiclass', average='micro', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()
                best_f1_micro        = F1Score(  task='multiclass', average='micro', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()

                best_precision_macro = Precision(task='multiclass', average='macro', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()
                best_recall_macro    = Recall(   task='multiclass', average='macro', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()
                best_f1_macro        = F1Score(  task='multiclass', average='macro', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()

                # ROC-AUC (OVR) on validation
                roc_auc_weighted = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                roc_auc_micro    = roc_auc_score(y_true, y_proba, multi_class='ovr', average='micro')
                roc_auc_macro    = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | NRF Loss: {loss2.item():.4f} | Val Acc: {acc:.4f}")

        epoch_logs.append(time.time() - start_time)

    print(f"Best validation acc: {best_acc:.4f} @ epoch {best_epoch}")

    # -------------------- Final evaluation on test set (semi-transductive/inductive) --------------------
    if args['final_evaluation']:
            print("Evaluating on test set using full graph...")
            neural_forest.load_state_dict(best_state_dict['ndf'])

            neural_forest.eval()

            with torch.no_grad():

                test_row_idx = torch.tensor(test_df.index.values, dtype=torch.long, device=device)

                feat_test = non_geo_features.index_select(0, test_row_idx)

                out_test = neural_forest(feat_test)     # shape: [#test_rows, n_class]

                pred_labels = out_test.argmax(dim=1)

                test_idx = torch.as_tensor(test_df.index.values, dtype=torch.long, device=device)
                y_test_true = y_nrf.index_select(0, test_idx)
                test_acc = (pred_labels == y_test_true).float().mean().item()
                y_true = y_test_true.detach().cpu().numpy()

                # Don't mask again — out_test is already test-only
                pred_proba = F.softmax(out_test, dim=1)

                y_pred = pred_labels.cpu().numpy()
                #y_proba = pred_proba.cpu().numpy()
                #y_true = y_nrf[row_test_mask].cpu().numpy()

                y_pred_decoded = [index_to_label[i] for i in y_pred]
                y_true_decoded = [index_to_label[i] for i in y_true]
    else:
        test_acc = best_acc
        y_pred_decoded = []
        y_true_decoded = []

    # Return everything you were already returning (so your file-writing stays intact)
    return (
        test_acc, best_epoch, best_precision, best_recall, best_f1,
        y_pred_decoded, y_true_decoded,
        best_precision_micro, best_recall_micro, best_f1_micro,
        best_precision_macro, best_recall_macro, best_f1_macro,
        roc_auc_weighted, roc_auc_micro, roc_auc_macro,
        epoch_logs
    )

