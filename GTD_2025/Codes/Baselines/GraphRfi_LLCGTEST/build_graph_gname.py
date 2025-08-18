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



class GCNForGname(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_class):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)  # node embeddings
        self.cls = torch.nn.Linear(hidden_channels, n_class)    # node-level gname logits

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)                       # (num_nodes, hidden_channels)
        logits = self.cls(h)                # (num_nodes, n_class)
        return h, logits


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
    data.edge_index_full = edge_index_full

    model = GCNForGname(
        in_channels=data.num_node_features,
        hidden_channels=args['embed_dim'],
        n_class=args['n_class']
    ).to(device)

    feat_layer = {
        "gtd100": GTD100FeatureLayer,
        "gtd200": GTD200FeatureLayer,
        "gtd300": GTD300FeatureLayer,
        "gtd478": GTD478FeatureLayer,
    }[args['partition']](dropout_rate=args['feat_dropout']).to(device)

    forest_in_dim = feat_layer.get_out_feature_size() + args['embed_dim']  # 1024 + D
    forest = Forest(
        n_tree=args['n_tree'],
        tree_depth=args['tree_depth'],
        n_in_feature=forest_in_dim,   # <-- 1024 + embed_dim
        tree_feature_rate=args['tree_feature_rate'],
        n_class=args['n_class']
    ).to(device)

    

    criterion_ndf = torch.nn.NLLLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(feat_layer.parameters()) + list(forest.parameters()),
        lr=args['lr']
    )

    ce = torch.nn.CrossEntropyLoss()

    # Move to device
    data = data.to(device)
    y_nrf = y_nrf.to(device)
    non_geo_features = non_geo_features.to(device)

    # Keep masks on CPU for indexing; ensure they’re CPU just in case
    train_mask_cpu = train_mask.cpu()
    val_mask_cpu   = val_mask.cpu()
    test_mask_cpu  = test_mask.cpu()

    # Use CPU index to build row-level masks, THEN move masks to device
    row_to_node_index_cpu = row_to_node_index.cpu()
    row_train_mask = train_mask_cpu[row_to_node_index_cpu].to(device)
    row_val_mask   = val_mask_cpu[row_to_node_index_cpu].to(device)
    row_test_mask  = test_mask_cpu[row_to_node_index_cpu].to(device)

    # ALSO keep a device copy for indexing GPU node embeddings/logits
    row_to_node_index_dev = row_to_node_index.to(device)

    best_acc, best_epoch, best_state_dict = -1.0, -1, None
    epoch_logs, no_improvement = [], 0
    patience = 300 if args['final_evaluation'] else 100

    for epoch in range(args['epochs']):
        start_time = time.time()
        model.train(); forest.train()
        optimizer.zero_grad()

        # ---- forward (train) ----
        h_nodes, gcn_node_logits = model(data.x, data.edge_index)
        h_rows = h_nodes[row_to_node_index_dev]                 # (B, embed_dim)
        gcn_row_logits = gcn_node_logits[row_to_node_index_dev]

        tab1024 = feat_layer(non_geo_features)                  # (B, 1024)
        fused = torch.cat([tab1024, h_rows], dim=1)             # (B, 1024 + embed_dim)
        out_probs = forest(fused)                               # (B, C)  probabilities

        loss_ndf = criterion_ndf(torch.log(out_probs[row_train_mask] + 1e-12),
                                y_nrf[row_train_mask])
        loss_gcn = ce(gcn_row_logits[row_train_mask], y_nrf[row_train_mask])
        loss = loss_ndf + loss_gcn
        
        loss.backward()
        optimizer.step()

        # ---- validation ----
        model.eval(); forest.eval()
        with torch.no_grad():
            h_nodes_v, gcn_node_logits_v = model(data.x, data.edge_index)
            h_rows_v = h_nodes_v[row_to_node_index_dev]
            gcn_row_probs_v = F.softmax(gcn_node_logits_v[row_to_node_index_dev], dim=1)

            tab1024_v = feat_layer(non_geo_features)
            fused_v = torch.cat([tab1024_v, h_rows_v], dim=1)
            out_probs_v = forest(fused_v)
            nrf_row_probs_v = F.softmax(out_probs_v, dim=1)

            # α search for ensemble
            alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
            best_a_epoch, best_val_acc_epoch, best_p_epoch = 0.5, -1.0, None
            for a in alphas:
                p = a * nrf_row_probs_v[row_val_mask] + (1 - a) * gcn_row_probs_v[row_val_mask]
                pred = p.argmax(dim=1)
                acc_a = (pred == y_nrf[row_val_mask]).float().mean().item()
                if acc_a > best_val_acc_epoch:
                    best_val_acc_epoch, best_a_epoch, best_p_epoch = acc_a, a, p

            if best_val_acc_epoch > best_acc:
                best_acc, best_epoch = best_val_acc_epoch, epoch
                best_state_dict = {'gcn': model.state_dict(),
                                   'ndf': forest.state_dict(),
                                   'alpha': best_a_epoch}
                no_improvement = 0

                # metrics on ensemble
                pred_labels = best_p_epoch.argmax(dim=1)
                y_pred_tensor = pred_labels
                y_true_tensor = y_nrf[row_val_mask]
                y_pred = y_pred_tensor.cpu().numpy()
                y_proba = best_p_epoch.cpu().numpy()
                y_true = y_true_tensor.cpu().numpy()

                ncls = args['n_class']
                best_precision = Precision(task='multiclass', average='weighted', num_classes=ncls).to(device)(y_pred_tensor, y_true_tensor).item()
                best_recall    = Recall(   task='multiclass', average='weighted', num_classes=ncls).to(device)(y_pred_tensor, y_true_tensor).item()
                best_f1        = F1Score(  task='multiclass', average='weighted', num_classes=ncls).to(device)(y_pred_tensor, y_true_tensor).item()
                best_precision_micro = Precision(task='multiclass', average='micro', num_classes=ncls).to(device)(y_pred_tensor, y_true_tensor).item()
                best_recall_micro    = Recall(   task='multiclass', average='micro', num_classes=ncls).to(device)(y_pred_tensor, y_true_tensor).item()
                best_f1_micro        = F1Score(  task='multiclass', average='micro', num_classes=ncls).to(device)(y_pred_tensor, y_true_tensor).item()
                best_precision_macro = Precision(task='multiclass', average='macro', num_classes=ncls).to(device)(y_pred_tensor, y_true_tensor).item()
                best_recall_macro    = Recall(   task='multiclass', average='macro', num_classes=ncls).to(device)(y_pred_tensor, y_true_tensor).item()
                best_f1_macro        = F1Score(  task='multiclass', average='macro', num_classes=ncls).to(device)(y_pred_tensor, y_true_tensor).item()

                roc_auc_weighted = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                roc_auc_micro    = roc_auc_score(y_true, y_proba, multi_class='ovr', average='micro')
                roc_auc_macro    = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss NDF: {loss_ndf.item():.4f} | Loss GCN: {loss_gcn.item():.4f} | Val Acc (ens): {best_val_acc_epoch:.4f}")

        epoch_logs.append(time.time() - start_time)

    print(f"Best validation acc: {best_acc:.4f} @ epoch {best_epoch}")

    # ---- test ----
    if args['final_evaluation']:
        print("Evaluating on test set using full graph...")
        model.load_state_dict(best_state_dict['gcn'])
        forest.load_state_dict(best_state_dict['ndf'])
        alpha = best_state_dict['alpha']

        model.eval(); forest.eval()
        with torch.no_grad():
            h_nodes_t, gcn_node_logits_t = model(data.x.to(device), data.edge_index_full.to(device))
            h_rows_t = h_nodes_t[row_to_node_index_dev]
            gcn_row_probs_t = F.softmax(gcn_node_logits_t[row_to_node_index_dev], dim=1)

            tab1024_t = feat_layer(non_geo_features)
            fused_t = torch.cat([tab1024_t, h_rows_t], dim=1)
            out_probs_t = forest(fused_t)
            nrf_row_probs_t = F.softmax(out_probs_t, dim=1)

            p_combo = alpha * nrf_row_probs_t[row_test_mask] + (1 - alpha) * gcn_row_probs_t[row_test_mask]
    
            pred_labels = p_combo.argmax(dim=1)
            pred_proba = p_combo
            test_acc = (pred_labels == y_nrf[row_test_mask]).float().mean().item()

            y_pred = pred_labels.cpu().numpy()
            y_proba = pred_proba.cpu().numpy()
            y_true = y_nrf[row_test_mask].cpu().numpy()
            y_pred_decoded = [index_to_label[i] for i in y_pred]
            y_true_decoded = [index_to_label[i] for i in y_true]
    else:
        test_acc = best_acc
        y_pred_decoded = []
        y_true_decoded = []

    return (test_acc, best_epoch, best_precision, best_recall, best_f1,
            y_pred_decoded, y_true_decoded,
            best_precision_micro, best_recall_micro, best_f1_micro,
            best_precision_macro, best_recall_macro, best_f1_macro,
            roc_auc_weighted, roc_auc_micro, roc_auc_macro,
            epoch_logs)