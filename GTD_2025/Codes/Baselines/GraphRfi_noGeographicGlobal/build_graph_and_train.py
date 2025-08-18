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
from torch_geometric.utils import add_self_loops




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


from sklearn.neighbors import NearestNeighbors

def build_graph_data_knn(df, label_index, continuous_col, k=5):
    # Define node identity by 4 key features
    node_keys = df[['attacktype1', 'target1']].astype(str).agg('_'.join, axis=1)
    df = df.copy()
    df['node_key'] = node_keys

    # Get unique node keys and mapping
    unique_keys = df['node_key'].unique()
    key_to_index = {k: i for i, k in enumerate(unique_keys)}
    N = len(key_to_index)

    # Node features = raw features (numerical)
    node_feats_df = df.drop_duplicates('node_key')[['attacktype1', 'target1']].astype(float)
    node_feats_df = node_feats_df.set_index(df['node_key'].drop_duplicates()).loc[unique_keys]
    x_np = node_feats_df.values
    x = torch.tensor(x_np, dtype=torch.float32)

    # ------------------------
    # Step: k-NN edge creation
    # ------------------------
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(x_np)
    distances, indices = nbrs.kneighbors(x_np)

    # Skip self-loops (first index is self)
    edge_index_list = []
    for src_node, neighbors in enumerate(indices):
        for tgt_node in neighbors[1:]:  # skip self
            edge_index_list.append([src_node, tgt_node])
            edge_index_list.append([tgt_node, src_node])  # make undirected

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).T  # shape (2, num_edges)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)

    # Regression targets for GCN
    y_gcn = torch.full((N,), -1.0, dtype=torch.float32)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    train_df, val_df, test_df = handle_leakage(df)

    assigned_nodes = set()

    for split_df, mask in zip([train_df, val_df, test_df], [train_mask, val_mask, test_mask]):
        for _, row in split_df.iterrows():
            idx = key_to_index[row['node_key']]
            if idx not in assigned_nodes:
                y_gcn[idx] = row[continuous_col]
                mask[idx] = True
                assigned_nodes.add(idx)

    # NRF input = full per-row feature vectors
    nrf_input_cols = [col for col in df.columns if col not in ['gname', 'node_key']]
    nrf_input = torch.tensor(df[nrf_input_cols].astype(float).fillna(0).values, dtype=torch.float32)

    # Mapping row to node
    row_to_node_index = torch.tensor(df['node_key'].map(key_to_index).values, dtype=torch.long)

    # NRF classification labels
    y_nrf = torch.tensor(df['gname'].map(label_index).values, dtype=torch.long)
    index_to_label = {v: k for k, v in label_index.items()}

    return Data(x=x, edge_index=edge_index), y_gcn, y_nrf, nrf_input, train_mask, val_mask, test_mask, row_to_node_index, index_to_label



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

def train_joint(data, y_gcn, y_nrf, non_geo_features, train_mask, val_mask, test_mask, args, row_to_node_index, index_to_label, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNRegressor(data.num_node_features, args['embed_dim']).to(device)

    # Depending on what partition
    if args['partition'] == "gtd100":
        feat_layer = GTD100FeatureLayer(dropout_rate=args['feat_dropout'])
    elif args['partition'] == "gtd200":
        feat_layer = GTD200FeatureLayer(dropout_rate=args['feat_dropout'])
    elif args['partition'] == "gtd300":
        feat_layer = GTD300FeatureLayer(dropout_rate=args['feat_dropout'])
    elif args['partition'] == "gtd478":
        feat_layer = GTD478FeatureLayer(dropout_rate=args['feat_dropout'])

    # Create forest object
    forest = Forest(
        n_tree=args['n_tree'],
        tree_depth=args['tree_depth'],
        n_in_feature=feat_layer.get_out_feature_size(),
        tree_feature_rate=args['tree_feature_rate'],
        n_class=args['n_class']
    )

    #Define ndf
    neural_forest = NeuralDecisionForest(feat_layer, forest).to(device)

    # Optimizer and GCN loss
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(neural_forest.parameters()), lr=args['lr'])
    mse_loss = torch.nn.MSELoss()

    # GPU Compatible
    data = data.to(device)
    y_gcn = y_gcn.to(device)
    non_geo_features = non_geo_features.to(device)
    y_nrf = y_nrf.clone().detach().to(torch.long).to(device)

    best_acc = -1
    best_epoch = -1

    row_train_mask = train_mask[row_to_node_index]
    row_val_mask = val_mask[row_to_node_index]
    row_test_mask = test_mask[row_to_node_index]

    epoch_logs = []
    if args['final_evaluation'] == True:
        patience = 300
    else:
        patience = 100
        
    no_improvement = 0
    best_state_dict = None

    epoch_iter = range(args['epochs'])
    #if verbose:
    #    epoch_iter = tqdm(epoch_iter, desc="Training epochs")

    for epoch in epoch_iter:
        start_time = time.time()
        #Train GCN and NRF
        model.train()
        neural_forest.train()

        #Zero gradients
        optimizer.zero_grad()

        #Predict GCN and calculate error
        pred = model(data.x, data.edge_index.to(device))
        pred_error = (pred - y_gcn).pow(2).unsqueeze(1)
        per_row_pred_error = pred_error[row_to_node_index]

        #Input to NRF
        input_features = torch.cat([non_geo_features, per_row_pred_error], dim=1)

        # Compute both losses
        loss1 = mse_loss(pred[train_mask], y_gcn[train_mask])
        out_forest = neural_forest(input_features)

        loss2 = neural_forest.loss(out_forest[row_train_mask], y_nrf[row_train_mask])
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        # Evaluate NRF
        model.eval()
        neural_forest.eval()
        with torch.no_grad():
            pred_eval = model(data.x, data.edge_index.to(device))
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

                no_improvement = 0  # reset counter

                y_pred_tensor = best_labels
                y_true_tensor = y_nrf[row_val_mask]

                y_pred = best_labels.cpu().numpy()
                y_proba = best_proba.cpu().numpy()
                y_true = y_nrf[row_val_mask].cpu().numpy()

                y_pred_decoded = [index_to_label[i] for i in y_pred]
                y_true_decoded = [index_to_label[i] for i in y_true]

                # Initialize and move metrics to device
                metric_precision = Precision(task='multiclass', average='weighted', num_classes=30).to(device)
                metric_recall = Recall(task='multiclass', average='weighted', num_classes=30).to(device)
                metric_f1 = F1Score(task='multiclass', average='weighted', num_classes=30).to(device)

                micro_precision = Precision(task='multiclass', average='micro', num_classes=30).to(device)
                micro_recall = Recall(task='multiclass', average='micro', num_classes=30).to(device)
                micro_f1 = F1Score(task='multiclass', average='micro', num_classes=30).to(device)

                macro_precision = Precision(task='multiclass', average='macro', num_classes=30).to(device)
                macro_recall = Recall(task='multiclass', average='macro', num_classes=30).to(device)
                macro_f1 = F1Score(task='multiclass', average='macro', num_classes=30).to(device)

                # Compute values
                best_precision = metric_precision(y_pred_tensor, y_true_tensor).item()
                best_recall = metric_recall(y_pred_tensor, y_true_tensor).item()
                best_f1 = metric_f1(y_pred_tensor, y_true_tensor).item()

                best_precision_micro = micro_precision(y_pred_tensor, y_true_tensor).item()
                best_recall_micro = micro_recall(y_pred_tensor, y_true_tensor).item()
                best_f1_micro = micro_f1(y_pred_tensor, y_true_tensor).item()

                best_precision_macro = macro_precision(y_pred_tensor, y_true_tensor).item()
                best_recall_macro = macro_recall(y_pred_tensor, y_true_tensor).item()
                best_f1_macro = macro_f1(y_pred_tensor, y_true_tensor).item()

                roc_auc_weighted = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                roc_auc_micro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='micro')
                roc_auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')

            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch+1:02d} | GCN MSE Loss: {loss1.item():.4f} | NRF Loss: {loss2.item():.4f} | JOINT Loss: {loss.item():.4f} | NRF Acc: {acc:.4f}")
        epoch_time = time.time() - start_time
        epoch_logs.append(epoch_time)

    print(f"Best acc/epoch: {best_acc}, epoch {best_epoch}")

    if args['final_evaluation']:
        print("Predicting on test set")

        # Load best weights
        model.load_state_dict(best_state_dict['gcn'])
        neural_forest.load_state_dict(best_state_dict['ndf'])

        model.eval()
        neural_forest.eval()

        with torch.no_grad():
            # Run GCN on the full graph
            gcn_pred = model(data.x.to(device), data.edge_index.to(device))
            pred_error = (gcn_pred - y_gcn).pow(2).unsqueeze(1)

            # Get per-row prediction error and full input to NRF
            per_row_pred_error = pred_error[row_to_node_index]
            input_features = torch.cat([non_geo_features, per_row_pred_error], dim=1)

            # Predict with NRF
            out_forest = neural_forest(input_features)
            pred_labels = out_forest[row_test_mask].argmax(dim=1)
            pred_proba = F.softmax(out_forest[row_test_mask], dim=1)

            # Accuracy
            test_acc = (pred_labels == y_nrf[row_test_mask]).float().mean().item()

            # Decode for return if needed
            y_pred = pred_labels.cpu().numpy()
            y_proba = pred_proba.cpu().numpy()
            y_true = y_nrf[row_test_mask].cpu().numpy()

            y_pred_decoded = [index_to_label[i] for i in y_pred]
            y_true_decoded = [index_to_label[i] for i in y_true]

    if not args['final_evaluation']:
        test_acc = best_acc

    return test_acc, best_epoch, best_precision, best_recall, best_f1, y_pred_decoded, y_true_decoded, best_precision_micro, best_recall_micro, best_f1_micro, best_precision_macro, best_recall_macro, best_f1_macro, roc_auc_weighted, roc_auc_micro, roc_auc_macro, epoch_logs