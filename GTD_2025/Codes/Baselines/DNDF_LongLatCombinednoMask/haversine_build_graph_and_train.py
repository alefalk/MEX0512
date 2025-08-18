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



def build_graph_data(df, label_index, continuous_col, dist_threshold_km=1.0):
    # Unique coordinates and mapping
    full_coords = df['longlat'].drop_duplicates().reset_index(drop=True)
    coord_to_index = {coord: i for i, coord in enumerate(full_coords)}
    coords = list(coord_to_index.keys())
    N = len(coords)

    # Node features
    x = torch.tensor(coords, dtype=torch.float32)

    train_df, val_df, test_df = handle_leakage(df)

    # Raw coordinates as NRF input
    longlat_coords = torch.tensor(df['longlat'].apply(lambda x: list(x)).to_list(), dtype=torch.float32)
    non_location_cols = [col for col in df.columns if col not in ['longlat', 'gname']]
    other_feats = torch.tensor(df[non_location_cols].astype(float).values, dtype=torch.float32)
    nrf_input = torch.cat([longlat_coords, other_feats], dim=1)

    row_to_node_index = torch.tensor(df['longlat'].map(coord_to_index).values, dtype=torch.long)
    y_nrf = torch.tensor(df['gname'].map(label_index).values, dtype=torch.long)
    index_to_label = {v: k for k, v in label_index.items()}

    return y_nrf, nrf_input, train_df, val_df, test_df, row_to_node_index, index_to_label

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

def train_joint(y_nrf, non_geo_features, args,
                row_to_node_index, index_to_label, verbose, train_df, val_df, test_df):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Select feature extractor
    feat_layer = {
        "gtd100": GTD100FeatureLayer,
        "gtd200": GTD200FeatureLayer,
        "gtd300": GTD300FeatureLayer,
        "gtd478": GTD478FeatureLayer,
    }[args['partition']](dropout_rate=args['feat_dropout'])

    # Create neural decision forest
    forest = Forest(
        n_tree=args['n_tree'],
        tree_depth=args['tree_depth'],
        n_in_feature=feat_layer.get_out_feature_size(),
        tree_feature_rate=args['tree_feature_rate'],
        n_class=args['n_class']
    )
    neural_forest = NeuralDecisionForest(feat_layer, forest).to(device)

    optimizer = torch.optim.AdamW(
        list(neural_forest.parameters()), lr=args['lr']
    )

    y_nrf = y_nrf.to(device)
    non_geo_features = non_geo_features.to(device)

    best_acc = -1
    best_epoch = -1
    best_state_dict = None
    epoch_logs = []
    no_improvement = 0

    if args['final_evaluation'] == True:
        patience = 300
    else:
        patience = 100
        
    for epoch in range(args['epochs']):
        start_time = time.time()

        neural_forest.train()
        optimizer.zero_grad()

        input_features_train = torch.cat([non_geo_features[train_df.index]], dim=1)
        labels_train = y_nrf[train_df.index]


        out_forest = neural_forest(input_features_train)
        loss2 = neural_forest.loss(out_forest, labels_train)
        loss = loss2

        loss.backward()
        optimizer.step()

        # Validation
        neural_forest.eval()
        with torch.no_grad():
            input_features_val = torch.cat([non_geo_features[val_df.index]], dim=1)
            labels_val = y_nrf[val_df.index]

            out_forest = neural_forest(input_features_val)
            pred_labels = out_forest.argmax(dim=1)
            pred_proba = F.softmax(out_forest, dim=1)
            acc = (pred_labels == labels_val).float().mean().item()

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_labels = pred_labels
                best_proba = pred_proba
                best_state_dict = {
                    'ndf': neural_forest.state_dict()
                }
                no_improvement = 0

                y_pred = pred_labels.cpu().numpy()
                y_proba = best_proba.cpu().numpy()
                y_true = labels_val.cpu().numpy()

                y_pred_tensor = pred_labels
                y_true_tensor = labels_val

                # Metrics
                best_precision = Precision(task='multiclass', average='weighted', num_classes=30).to(device)(y_pred_tensor, y_true_tensor).item()
                best_recall = Recall(task='multiclass', average='weighted', num_classes=30).to(device)(y_pred_tensor, y_true_tensor).item()
                best_f1 = F1Score(task='multiclass', average='weighted', num_classes=30).to(device)(y_pred_tensor, y_true_tensor).item()

                best_precision_micro = Precision(task='multiclass', average='micro', num_classes=30).to(device)(y_pred_tensor, y_true_tensor).item()
                best_recall_micro = Recall(task='multiclass', average='micro', num_classes=30).to(device)(y_pred_tensor, y_true_tensor).item()
                best_f1_micro = F1Score(task='multiclass', average='micro', num_classes=30).to(device)(y_pred_tensor, y_true_tensor).item()

                best_precision_macro = Precision(task='multiclass', average='macro', num_classes=30).to(device)(y_pred_tensor, y_true_tensor).item()
                best_recall_macro = Recall(task='multiclass', average='macro', num_classes=30).to(device)(y_pred_tensor, y_true_tensor).item()
                best_f1_macro = F1Score(task='multiclass', average='macro', num_classes=30).to(device)(y_pred_tensor, y_true_tensor).item()

                roc_auc_weighted = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                roc_auc_micro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='micro')
                roc_auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | NRF Loss: {loss2.item():.4f} | Joint: {loss.item():.4f} | Val Acc: {acc:.4f}")

        epoch_logs.append(time.time() - start_time)

    print(f"Best validation acc: {best_acc:.4f} @ epoch {best_epoch}")

    # Final evaluation on test set
    if args['final_evaluation']:
        print("Evaluating on test set using full graph...")
        neural_forest.load_state_dict(best_state_dict['ndf'])

        neural_forest.eval()

        with torch.no_grad():
            input_features_test = torch.cat([non_geo_features[test_df.index]], dim=1)
            labels_test = y_nrf[test_df.index]

            out_forest = neural_forest(input_features_test)
            pred_labels = out_forest.argmax(dim=1)
            pred_proba = F.softmax(out_forest, dim=1)

            test_acc = (pred_labels == labels_test).float().mean().item()

            y_pred = pred_labels.cpu().numpy()
            y_proba = pred_proba.cpu().numpy()
            y_true = labels_test = y_nrf[test_df.index].cpu().numpy()

            y_pred_decoded = [index_to_label[i] for i in y_pred]
            y_true_decoded = [index_to_label[i] for i in y_true]
    else:
        test_acc = best_acc
        y_pred_decoded = [index_to_label[i] for i in best_labels.cpu().numpy()]
        y_true_decoded = [index_to_label[i] for i in labels_val.cpu().numpy()]

    return (
        test_acc, best_epoch, best_precision, best_recall, best_f1,
        y_pred_decoded, y_true_decoded,
        best_precision_micro, best_recall_micro, best_f1_micro,
        best_precision_macro, best_recall_macro, best_f1_macro,
        roc_auc_weighted, roc_auc_micro, roc_auc_macro,
        epoch_logs
    )