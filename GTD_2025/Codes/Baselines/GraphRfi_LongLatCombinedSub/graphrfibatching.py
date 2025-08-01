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
from torch_geometric.loader import DataLoader



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


def build_graph_data(df, label_index, continuous_col):
    # Step 1: Build graph (GCN input)
    full_coords = df['longlat'].drop_duplicates().reset_index(drop=True)
    coord_to_index = {coord: i for i, coord in enumerate(full_coords)}
    N = len(coord_to_index)

    adj = np.eye(N)
    edge_index = torch.tensor(np.vstack(coo_matrix(adj).nonzero()), dtype=torch.long)
    x = torch.tensor(np.array(list(coord_to_index.keys())), dtype=torch.float32)

    # Step 2: Initialize GCN targets and masks
    y_gcn = torch.full((N,), -1.0, dtype=torch.float32)
    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    train_df, test_df = handle_leakage(df)

    for _, row in train_df.iterrows():
        coord = row['longlat']
        if coord in coord_to_index:
            idx = coord_to_index[coord]
            y_gcn[idx] = row[continuous_col]
            train_mask[idx] = True

    for _, row in test_df.iterrows():
        coord = row['longlat']
        if coord in coord_to_index:
            idx = coord_to_index[coord]
            y_gcn[idx] = row[continuous_col]
            test_mask[idx] = True

    # Step 3: Use raw longlat values as numeric features
    longlat_coords = torch.tensor(df['longlat'].apply(lambda x: list(x)).to_list(), dtype=torch.float32)  # shape: (N, 2)

    # Step 4: Build non-location features (full df)
    non_location_cols = [col for col in df.columns if col not in ['longlat', 'gname']]
    other_feats = torch.tensor(df[non_location_cols].astype(float).values, dtype=torch.float32)

    # Step 5: Concatenate longlat coordinates + other features for NRF input
    nrf_input = torch.cat([longlat_coords, other_feats], dim=1)

    # Step 6: Map each row to a node
    row_to_node_index = torch.tensor(df['longlat'].map(coord_to_index).values, dtype=torch.long)

    # Step 7: Create NRF labels (one per row)
    y_nrf = torch.tensor(df['gname'].map(label_index).values, dtype=torch.long)

    index_to_label = {v: k for k, v in label_index.items()}

    data_list = []
    for i, row in df.iterrows():
        node_feat = torch.tensor(list(row['longlat']), dtype=torch.float32).unsqueeze(0)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # self-loop
        data = Data(x=node_feat, edge_index=edge_index, idx=torch.tensor(row_to_node_index[i]))
        data_list.append(data)

    return data_list, y_gcn, y_nrf, nrf_input, train_mask, test_mask, row_to_node_index, index_to_label

    #return Data(x=x, edge_index=edge_index), y_gcn, y_nrf, nrf_input, train_mask, test_mask, row_to_node_index, index_to_label

def handle_leakage(df):
    train_frames = []
    test_frames = []
    for _, group in df.groupby('gname'):
        split = int(len(group) * 0.7)
        train_frames.append(group.iloc[:split])
        test_frames.append(group.iloc[split:])
    return shuffle(pd.concat(train_frames)), shuffle(pd.concat(test_frames))

def train_joint(data_list, y_gcn, y_nrf, non_geo_features, train_mask, test_mask, args, row_to_node_index, index_to_label, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use raw longlat features (2D) as GCN input
    model = GCNRegressor(in_channels=2, hidden_channels=args['embed_dim']).to(device)

    # Feature extractor based on partition
    if args['partition'] == "gtd100":
        feat_layer = GTD100FeatureLayer(dropout_rate=args['feat_dropout'])
    elif args['partition'] == "gtd200":
        feat_layer = GTD200FeatureLayer(dropout_rate=args['feat_dropout'])
    elif args['partition'] == "gtd300":
        feat_layer = GTD300FeatureLayer(dropout_rate=args['feat_dropout'])
    elif args['partition'] == "gtd478":
        feat_layer = GTD478FeatureLayer(dropout_rate=args['feat_dropout'])

    # Mask rows based on node indices
    row_train_mask = train_mask[row_to_node_index]
    row_test_mask = test_mask[row_to_node_index]

    train_indices = torch.where(row_train_mask)[0]
    test_indices = torch.where(row_test_mask)[0]

    train_data = [data_list[i.item()] for i in train_indices]
    test_data = [data_list[i.item()] for i in test_indices]

    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)

    # Neural decision forest setup
    forest = Forest(
        n_tree=args['n_tree'],
        tree_depth=args['tree_depth'],
        n_in_feature=feat_layer.get_out_feature_size(),
        tree_feature_rate=args['tree_feature_rate'],
        n_class=args['n_class']
    )
    neural_forest = NeuralDecisionForest(feat_layer, forest).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(neural_forest.parameters()), lr=args['lr'])
    mse_loss = torch.nn.MSELoss()

    # Move tensors to device
    y_gcn = y_gcn.to(device)
    y_nrf = y_nrf.to(torch.long).to(device)
    non_geo_features = non_geo_features.to(device)

    best_acc = -1
    best_epoch = -1
    best_state_dict = None
    epoch_logs = []
    patience = 300
    no_improvement = 0

    epoch_iter = tqdm(range(args['epochs']), desc="Training epochs") if verbose else range(args['epochs'])

    for epoch in epoch_iter:
        model.train()
        neural_forest.train()
        epoch_loss1, epoch_loss2 = 0.0, 0.0
        start_time = time.time()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch.x, batch.edge_index)
            batch_indices = batch.idx
            gcn_targets = y_gcn[batch_indices]
            nrf_targets = y_nrf[batch_indices]
            other_feats_batch = non_geo_features[batch_indices]

            pred_error = (pred - gcn_targets).pow(2).unsqueeze(1)
            input_features = torch.cat([other_feats_batch, pred_error], dim=1)

            out_forest = neural_forest(input_features)
            loss1 = mse_loss(pred, gcn_targets)
            loss2 = neural_forest.loss(out_forest, nrf_targets)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()

        # --- Evaluation ---
        model.eval()
        neural_forest.eval()

        all_preds, all_probs, all_targets = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index)

                batch_indices = batch.idx
                gcn_targets = y_gcn[batch_indices]
                nrf_targets = y_nrf[batch_indices]
                other_feats_batch = non_geo_features[batch_indices]

                pred_error = (pred - gcn_targets).pow(2).unsqueeze(1)
                input_features = torch.cat([other_feats_batch, pred_error], dim=1)

                out_forest = neural_forest(input_features)
                probs = F.softmax(out_forest, dim=1)
                pred_labels = out_forest.argmax(dim=1)

                all_preds.append(pred_labels)
                all_probs.append(probs)
                all_targets.append(nrf_targets)

        y_pred_tensor = torch.cat(all_preds)
        y_proba = torch.cat(all_probs)
        y_true_tensor = torch.cat(all_targets)
        acc = (y_pred_tensor == y_true_tensor).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_labels = y_pred_tensor
            best_proba = y_proba
            best_state_dict = {
                'gcn': model.state_dict(),
                'ndf': neural_forest.state_dict()
            }
            no_improvement = 0

            y_pred_np = best_labels.cpu().numpy()
            y_proba_np = best_proba.cpu().numpy()
            y_true_np = y_true_tensor.cpu().numpy()

            y_pred_decoded = [index_to_label[i] for i in y_pred_np]
            y_true_decoded = [index_to_label[i] for i in y_true_np]

            # TorchMetrics
            metric_args = dict(task='multiclass', num_classes=args['n_class'])
            precision = Precision(average='weighted', **metric_args).to(device)(best_labels, y_true_tensor).item()
            recall = Recall(average='weighted', **metric_args).to(device)(best_labels, y_true_tensor).item()
            f1 = F1Score(average='weighted', **metric_args).to(device)(best_labels, y_true_tensor).item()

            precision_micro = Precision(average='micro', **metric_args).to(device)(best_labels, y_true_tensor).item()
            recall_micro = Recall(average='micro', **metric_args).to(device)(best_labels, y_true_tensor).item()
            f1_micro = F1Score(average='micro', **metric_args).to(device)(best_labels, y_true_tensor).item()

            precision_macro = Precision(average='macro', **metric_args).to(device)(best_labels, y_true_tensor).item()
            recall_macro = Recall(average='macro', **metric_args).to(device)(best_labels, y_true_tensor).item()
            f1_macro = F1Score(average='macro', **metric_args).to(device)(best_labels, y_true_tensor).item()

            roc_auc_weighted = roc_auc_score(y_true_np, y_proba_np, multi_class='ovr', average='weighted')
            roc_auc_micro = roc_auc_score(y_true_np, y_proba_np, multi_class='ovr', average='micro')
            roc_auc_macro = roc_auc_score(y_true_np, y_proba_np, multi_class='ovr', average='macro')

        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        avg_loss1 = epoch_loss1 / len(train_loader)
        avg_loss2 = epoch_loss2 / len(train_loader)
        avg_joint = avg_loss1 + avg_loss2

        if verbose:
            print(f"Epoch {epoch+1:02d} | GCN MSE Loss: {avg_loss1:.4f} | NRF Loss: {avg_loss2:.4f} | JOINT Loss: {avg_joint:.4f} | NRF Acc: {acc:.4f}")

        epoch_logs.append(time.time() - start_time)

    print(f"Best acc/epoch: {best_acc:.4f}, epoch {best_epoch}")

    return (
        best_acc,
        best_epoch,
        precision, recall, f1,
        y_pred_decoded, y_true_decoded,
        precision_micro, recall_micro, f1_micro,
        precision_macro, recall_macro, f1_macro,
        roc_auc_weighted, roc_auc_micro, roc_auc_macro,
        epoch_logs
    )
