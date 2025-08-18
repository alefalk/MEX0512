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



class GCNEmbedder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)  # embeddings
        self.weap_head = torch.nn.Linear(hidden_channels, 1)    # optional: supervise with weaptype1

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)                           # (num_nodes, hidden_channels)
        weap_pred = self.weap_head(h).squeeze(-1)  # (num_nodes,)
        return h, weap_pred

def haversine_edges(coords, threshold_km=1.0):

    edges = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if haversine(coords[i], coords[j], Unit.KILOMETERS) <= threshold_km:
                edges.append((i, j))
                edges.append((j, i))
    return edges

def build_graph_data(df, label_index, continuous_col, dist_threshold_km=1.0):
    # Unique coordinates and mapping
    full_coords = df['longlat'].drop_duplicates().reset_index(drop=True)
    coord_to_index = {coord: i for i, coord in enumerate(full_coords)}
    coords = list(coord_to_index.keys())
    N = len(coords)

    # Edge index based on Haversine distance
    spatial_edges = haversine_edges(coords, threshold_km=dist_threshold_km)
    edge_index_full = torch.tensor(spatial_edges, dtype=torch.long).t().contiguous()

    # Node features
    x = torch.tensor(coords, dtype=torch.float32)

    # Targets and masks
    y_gcn = torch.full((N,), -1.0, dtype=torch.float32)
    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)

    train_df, val_df, test_df = handle_leakage(df)

    def mark(df_split, mask):
        for _, row in df_split.iterrows():
            coord = row['longlat']
            if coord in coord_to_index:
                idx = coord_to_index[coord]
                y_gcn[idx] = row[continuous_col]
                mask[idx] = True

    mark(train_df, train_mask)
    mark(val_df, val_mask)
    mark(test_df, test_mask)

    # Filter edges for training (only edges within train+val)
    allowed_nodes = (train_mask | val_mask).nonzero(as_tuple=True)[0].tolist()
    allowed_set = set(allowed_nodes)
    edge_index_train = torch.tensor([
        [src, dst] for src, dst in spatial_edges if src in allowed_set and dst in allowed_set
    ], dtype=torch.long).t().contiguous()

    # Raw coordinates as NRF input
    longlat_coords = torch.tensor(df['longlat'].apply(lambda x: list(x)).to_list(), dtype=torch.float32)
    non_location_cols = [col for col in df.columns if col not in ['longlat', 'gname']]
    other_feats = torch.tensor(df[non_location_cols].astype(float).values, dtype=torch.float32)
    nrf_input = torch.cat([longlat_coords, other_feats], dim=1)

    row_to_node_index = torch.tensor(df['longlat'].map(coord_to_index).values, dtype=torch.long)
    y_nrf = torch.tensor(df['gname'].map(label_index).values, dtype=torch.long)
    index_to_label = {v: k for k, v in label_index.items()}

    return Data(x=x, edge_index=edge_index_train), edge_index_full, y_gcn, y_nrf, nrf_input, train_mask, val_mask, test_mask, row_to_node_index, index_to_label

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

def train_joint(data, edge_index_full, y_gcn, y_nrf, non_geo_features,
                train_mask, val_mask, test_mask, args,
                row_to_node_index, index_to_label, verbose):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Store full edge index on data for convenience
    data.edge_index_full = edge_index_full

    model = GCNEmbedder(in_channels=data.num_node_features, hidden_channels=args['embed_dim']).to(device)

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
        list(model.parameters()) + list(neural_forest.parameters()),
        lr=args['lr']
    )
    mse_loss = torch.nn.MSELoss()

    # Move tensors to device
    data = data.to(device)
    y_gcn = y_gcn.to(device)
    y_nrf = y_nrf.to(device)
    non_geo_features = non_geo_features.to(device)

    row_train_mask = train_mask[row_to_node_index]
    row_val_mask = val_mask[row_to_node_index]
    row_test_mask = test_mask[row_to_node_index]

    best_acc, best_epoch, best_state_dict = -1, -1, None
    epoch_logs, no_improvement = [], 0
    patience = 300 if args['final_evaluation'] else 100
        
    for epoch in range(args['epochs']):
        start_time = time.time()

        model.train()
        neural_forest.train()
        optimizer.zero_grad()

        # ---- GCN forward on TRAIN graph ----
        h_nodes, weap_pred_nodes = model(data.x, data.edge_index)  # use training graph edge_index

        # Map node embeddings to rows
        h_rows = h_nodes[row_to_node_index]                        # (num_rows, embed_dim)

        # Concatenate embeddings to NRF inputs
        nrf_in = torch.cat([non_geo_features, h_rows], dim=1)

        # Losses
        loss_weap = mse_loss(weap_pred_nodes[train_mask], y_gcn[train_mask])   # node-level
        out_forest = neural_forest(nrf_in)                                     # row-level
        loss_ndf = neural_forest.loss(out_forest[row_train_mask], y_nrf[row_train_mask])

        loss = loss_weap + loss_ndf
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        neural_forest.eval()
        with torch.no_grad():
            # (Optionally you could run with data.edge_index_full; usually not needed for val)
            h_nodes_val, _ = model(data.x, data.edge_index)
            h_rows_val = h_nodes_val[row_to_node_index]
            nrf_in_val = torch.cat([non_geo_features, h_rows_val], dim=1)

            out_val = neural_forest(nrf_in_val)
            pred_labels = out_val[row_val_mask].argmax(dim=1)
            pred_proba = F.softmax(out_val[row_val_mask], dim=1)
            acc = (pred_labels == y_nrf[row_val_mask]).float().mean().item()

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_labels = pred_labels
                best_proba = pred_proba
                best_state_dict = {
                    'gcn': model.state_dict(),
                    'ndf': neural_forest.state_dict()
                }
                no_improvement = 0

                y_pred = pred_labels.cpu().numpy()
                y_proba = best_proba.cpu().numpy()
                y_true = y_nrf[row_val_mask].cpu().numpy()

                y_pred_tensor = pred_labels
                y_true_tensor = y_nrf[row_val_mask]

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
            print(f"Epoch {epoch:03d} | GCN Loss: {loss1.item():.4f} | NRF Loss: {loss2.item():.4f} | Joint: {loss.item():.4f} | Val Acc: {acc:.4f}")

        epoch_logs.append(time.time() - start_time)

    print(f"Best validation acc: {best_acc:.4f} @ epoch {best_epoch}")

    # Final evaluation on test set
    if args['final_evaluation']:
        print("Evaluating on test set using full graph...")
        model.load_state_dict(best_state_dict['gcn'])
        neural_forest.load_state_dict(best_state_dict['ndf'])

        model.eval()
        neural_forest.eval()

        with torch.no_grad():
            # use full edges if you want transductive reach at test
            h_nodes_test, _ = model(data.x.to(device), data.edge_index_full.to(device))
            h_rows_test = h_nodes_test[row_to_node_index]
            nrf_in_test = torch.cat([non_geo_features, h_rows_test], dim=1)

            out_test = neural_forest(nrf_in_test)
            pred_labels = out_test[row_test_mask].argmax(dim=1)
            pred_proba = F.softmax(out_test[row_test_mask], dim=1)
            
            test_acc = (pred_labels == y_nrf[row_test_mask]).float().mean().item()

            y_pred = pred_labels.cpu().numpy()
            y_proba = pred_proba.cpu().numpy()
            y_true = y_nrf[row_test_mask].cpu().numpy()

            y_pred_decoded = [index_to_label[i] for i in y_pred]
            y_true_decoded = [index_to_label[i] for i in y_true]
    else:
        test_acc = best_acc
        y_pred_decoded = [index_to_label[i] for i in best_labels.cpu().numpy()]
        y_true_decoded = [index_to_label[i] for i in y_nrf[row_val_mask].cpu().numpy()]

    return (
        test_acc, best_epoch, best_precision, best_recall, best_f1,
        y_pred_decoded, y_true_decoded,
        best_precision_micro, best_recall_micro, best_f1_micro,
        best_precision_macro, best_recall_macro, best_f1_macro,
        roc_auc_weighted, roc_auc_micro, roc_auc_macro,
        epoch_logs
    )