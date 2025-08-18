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



class GCNRegressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()

from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from sklearn.neighbors import NearestNeighbors

def _laplace_freq(counts, n_classes, alpha=1.0):
    # counts: np.array shape (n_classes,)
    smoothed = counts + alpha
    return smoothed / smoothed.sum()

def _build_node_splits(df, node_col, split_col='split'):
    """
    Ensures each node belongs to exactly one of {train,val,test}.
    Assumes df[split_col] already exists ('train'/'val'/'test') from your handle_leakage.
    """
    # First occurrence decides the node's split
    first_split = (
        df[[node_col, split_col]]
        .drop_duplicates(subset=[node_col], keep='first')
        .set_index(node_col)[split_col]
    )
    return first_split  # Series: node_key -> split

def build_graph_data_with_aggregates(
    df, label_index, continuous_col='weaptype1', k=5,
    weap_num_classes=None, atk_num_classes=None, tgt_num_classes=None,
    alpha=1.0
):
    # Prepare splits at row level (your handle_leakage)
    train_df, val_df, test_df = handle_leakage(df)
    train_df = train_df.copy(); train_df['split'] = 'train'
    val_df   = val_df.copy();   val_df['split'] = 'val'
    test_df  = test_df.copy();  test_df['split'] = 'test'
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Node key
    df['node_key'] = df['encodedlonglat']  # each unique encoded coord is a node
    unique_nodes = df['node_key'].drop_duplicates().tolist()
    node_to_idx = {k:i for i,k in enumerate(unique_nodes)}
    N = len(unique_nodes)

    # Ensure node-level split exclusivity
    node_split = _build_node_splits(df, node_col='node_key', split_col='split')

    # Determine class cardinalities if not provided
    if weap_num_classes is None:
        weap_num_classes = int(df['weaptype1'].max()) + 1
    if atk_num_classes is None:
        atk_num_classes = int(df['attacktype1'].max()) + 1
    if tgt_num_classes is None:
        tgt_num_classes = int(df['target1'].max()) + 1

    # Compute train-only node aggregates
    train_only = df[df['split'] == 'train']

    # counts per node
    grp = train_only.groupby('node_key')
    count = grp.size().reindex(unique_nodes).fillna(0).astype(int).values
    nkill_mean = grp['nkill'].mean().reindex(unique_nodes).fillna(0).values
    nkill_std  = grp['nkill'].std().reindex(unique_nodes).fillna(0).values
    nkill_med  = grp['nkill'].median().reindex(unique_nodes).fillna(0).values

    # categorical priors (Laplace-smoothed distributions)
    def cat_prior(col, n_classes):
        counts_mat = np.zeros((N, n_classes), dtype=float)
        # count occurrences per node/category
        for node, sub in grp:
            i = node_to_idx[node]
            vc = sub[col].value_counts().sort_index()
            idxs = vc.index.values.astype(int)
            counts_mat[i, idxs] = vc.values
        # smooth + normalize
        probs = np.apply_along_axis(lambda r: _laplace_freq(r, n_classes, alpha), 1, counts_mat)
        return probs

    weap_prior = cat_prior('weaptype1', weap_num_classes)
    atk_prior  = cat_prior('attacktype1', atk_num_classes)
    tgt_prior  = cat_prior('target1', tgt_num_classes)

    # Node features X = [count, nkill stats, atk_prior, tgt_prior, weap_prior]
    X = np.concatenate([
        count.reshape(-1,1),
        nkill_mean.reshape(-1,1),
        nkill_std.reshape(-1,1),
        nkill_med.reshape(-1,1),
        atk_prior, tgt_prior, weap_prior
    ], axis=1)
    X = torch.tensor(X, dtype=torch.float32)

    # k-NN edges on X
    nbrs = NearestNeighbors(n_neighbors=min(k+1, N)).fit(X.numpy())
    _, knn_idx = nbrs.kneighbors(X.numpy())
    edges = []
    for i, neigh in enumerate(knn_idx):
        for j in neigh[1:]:
            edges.append([i,j]); edges.append([j,i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)

    # y_gcn = train-mean(weaptype1) per node; fallback to global mean
    global_weap_mean = train_only['weaptype1'].mean() if len(train_only) else 0.0
    y_gcn = torch.full((N,), float(global_weap_mean), dtype=torch.float32)
    has_train = np.zeros(N, dtype=bool)
    for node, sub in grp:
        i = node_to_idx[node]
        y_gcn[i] = float(sub[continuous_col].mean())
        has_train[i] = True

    # masks (node-level, exclusive)
    train_mask = torch.tensor([node_split.get(n, 'test') == 'train' for n in unique_nodes], dtype=torch.bool)
    val_mask   = torch.tensor([node_split.get(n, 'test') == 'val'   for n in unique_nodes], dtype=torch.bool)
    test_mask  = torch.tensor([node_split.get(n, 'test') == 'test'  for n in unique_nodes], dtype=torch.bool)

    # row_to_node_index and NDF labels
    row_to_node_index = torch.tensor(df['node_key'].map(node_to_idx).values, dtype=torch.long)
    y_nrf = torch.tensor(df['gname'].map(label_index).values, dtype=torch.long)
    index_to_label = {v:k for k,v in label_index.items()}

    # Per-row non-geo features (don’t include raw encodedlonglat; it lost spatial meaning)
    nrf_input_cols = [c for c in df.columns if c not in ['gname','node_key', 'split']]
    non_geo_features = torch.tensor(df[nrf_input_cols].astype(float).fillna(0).values, dtype=torch.float32)

    data = Data(x=X, edge_index=edge_index)

    return data, y_gcn, y_nrf, non_geo_features, train_mask, val_mask, test_mask, row_to_node_index, index_to_label


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

def train_joint(data, y_gcn, y_nrf, non_geo_features,
                train_mask, val_mask, test_mask, args,
                row_to_node_index, index_to_label, verbose):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GCNRegressor(data.num_node_features, args['embed_dim']).to(device)

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
        list(model.parameters()) + list(neural_forest.parameters()), lr=args['lr']
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

        model.train()
        neural_forest.train()
        optimizer.zero_grad()

        # GCN prediction (on training graph)
        pred = model(data.x, data.edge_index)
        pred_error = (pred - y_gcn).pow(2).unsqueeze(1)
        per_row_pred_error = pred_error[row_to_node_index]

        input_features = torch.cat([non_geo_features, per_row_pred_error], dim=1)

        loss1 = mse_loss(pred[train_mask], y_gcn[train_mask])
        out_forest = neural_forest(input_features)
        loss2 = neural_forest.loss(out_forest[row_train_mask], y_nrf[row_train_mask])
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        neural_forest.eval()
        with torch.no_grad():
            pred_eval = model(data.x, data.edge_index)
            pred_error = (pred_eval - y_gcn).pow(2).unsqueeze(1)
            per_row_pred_error = pred_error[row_to_node_index]
            input_features = torch.cat([non_geo_features, per_row_pred_error], dim=1)

            out_forest = neural_forest(input_features)
            pred_labels = out_forest[row_val_mask].argmax(dim=1)
            pred_proba = F.softmax(out_forest[row_val_mask], dim=1)
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
            gcn_pred = model(data.x.to(device), data.edge_index.to(device))
            pred_error = (gcn_pred - y_gcn).pow(2).unsqueeze(1)
            per_row_pred_error = pred_error[row_to_node_index]
            input_features = torch.cat([non_geo_features, per_row_pred_error], dim=1)

            out_forest = neural_forest(input_features)
            pred_labels = out_forest[row_test_mask].argmax(dim=1)
            pred_proba = F.softmax(out_forest[row_test_mask], dim=1)

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