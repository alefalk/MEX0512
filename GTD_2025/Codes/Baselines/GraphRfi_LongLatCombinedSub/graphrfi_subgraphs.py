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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from haversine import haversine, Unit
from sklearn.model_selection import train_test_split




class GCNRegressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, batch):
        x = F.tanh(self.conv1(batch.x, batch.edge_index))
        x = self.conv2(x, batch.edge_index)

        center_embeddings = x[batch.center]
        out = self.out(center_embeddings)
        return out.squeeze()

def get_subgraph(center_id, edge_index, x_global):
    # Get neighbors (indices) of center node
    neighbors = edge_index[1][edge_index[0] == center_id]
    node_ids = torch.cat([torch.tensor([center_id]), neighbors]).unique()

    # Remap node indices locally
    id_map = {old_id.item(): i for i, old_id in enumerate(node_ids)}
    new_edges = []
    for source, destination in zip(*edge_index):
        if source in node_ids and destination in node_ids:
            new_edges.append((id_map[source.item()], id_map[destination.item()]))

    # If no edges exist, add a self-loop
    if len(new_edges) == 0:
        center_local_idx = 0  # only node in subgraph
        new_edges = [(0, 0)]
    else:
        center_local_idx = id_map[center_id.item()]

    sub_x = x_global[node_ids]
    sub_edge_index = torch.tensor(new_edges).T

    return sub_x, sub_edge_index, center_local_idx

def build_graph_data(traindata, testdata, continuous_col):
    combined = pd.concat([traindata, testdata], axis = 0)
    # Extract unique locations for node creation
    combined['location'] = list(zip(combined['longitude'], combined['latitude']))
    unique_locations = combined['location'].drop_duplicates().reset_index(drop=True)

    # Map locations to an identity
    location2id = {loc: idx for idx, loc in enumerate(unique_locations)}
    combined['location_id'] = combined['location'].map(location2id)

    # Encode labels
    le = LabelEncoder()
    combined['label'] = le.fit_transform(combined['gname'])

    # Get global node features
    coords = np.array([list(loc) for loc in unique_locations])  # [1790, 2]
    print("Feature Matrix shape: ", coords.shape)

    # Standardize features
    scaler = StandardScaler()
    x_global = scaler.fit_transform(coords)  # standardized features

    # Build global edge list using 1km Haversine
    edges = []
    coords_latlon = [(lat, lon) for lon, lat in unique_locations]
    for i in range(len(coords_latlon)):
        for j in range(i + 1, len(coords_latlon)):
            if haversine(coords_latlon[i], coords_latlon[j], Unit.KILOMETERS) <= 1.0:
                edges.append((i, j))
                edges.append((j, i))

    global_edge_index = torch.tensor(edges, dtype=torch.long).T  # shape [2, num_edges]

    traindata_list = []
    for row_idx, row in traindata.iterrows():
        center_id = location2id[(row['longitude'], row['latitude'])]
        label = le.transform([row['gname']])[0]
        
        x, edge_index, center_idx = get_subgraph(torch.tensor(center_id), global_edge_index, torch.tensor(x_global, dtype=torch.float))
        
        traindata_obj = Data(
        x=x, 
        edge_index=edge_index, 
        y=torch.tensor(label), 
        center=center_idx,
        idx=torch.tensor([row_idx])
    )

        traindata_list.append(traindata_obj)

    test_data_list = []
    for row_idx, row in testdata.iterrows():
        loc = (row['longitude'], row['latitude'])
        
        # Skip if location not in mapping (just in case)
        if loc not in location2id:
            continue
        
        center_id = location2id[loc]
        label = le.transform([row['gname']])[0]
        
        testdata_obj = Data(
        x=x, 
        edge_index=edge_index, 
        y=torch.tensor(label), 
        center=center_idx,
        idx=torch.tensor([row_idx])
    )

        test_data_list.append(testdata_obj)

    y_gcn = torch.tensor(combined[continuous_col].values, dtype=torch.float32)
    y_nrf = torch.tensor(combined['label'].values, dtype=torch.long)
    other_feats = combined.drop(columns=['gname', 'label', 'location', 'location_id'])
    nrf_input = torch.tensor(other_feats.astype(float).values, dtype=torch.float32)

    index_to_label = dict(enumerate(le.classes_))

    return traindata_list, test_data_list, y_gcn, y_nrf, nrf_input, index_to_label

from torch_geometric.loader import DataLoader

# BYGG UPP INPUT
def train_joint_subgraph(
    train_data_list, 
    test_data_list, 
    y_gcn, 
    y_nrf, 
    nrf_input, 
    args, 
    index_to_label, 
    verbose=True
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1: Initialize GCN regressor
    model = GCNRegressor(in_channels=2, hidden_channels=args['embed_dim']).to(device)

    # Step 2: Initialize feature layer and forest for NRF
    if args['partition'] == "gtd100":
        feat_layer = GTD100FeatureLayer(dropout_rate=args['feat_dropout'])
    elif args['partition'] == "gtd200":
        feat_layer = GTD200FeatureLayer(dropout_rate=args['feat_dropout'])
    elif args['partition'] == "gtd300":
        feat_layer = GTD300FeatureLayer(dropout_rate=args['feat_dropout'])
    elif args['partition'] == "gtd478":
        feat_layer = GTD478FeatureLayer(dropout_rate=args['feat_dropout'])

    forest = Forest(
        n_tree=args['n_tree'],
        tree_depth=args['tree_depth'],
        n_in_feature=feat_layer.get_out_feature_size(),
        tree_feature_rate=args['tree_feature_rate'],
        n_class=args['n_class']
    )

    neural_forest = NeuralDecisionForest(feat_layer, forest).to(device)

    subgraph_indices = [data.idx.item() for data in train_data_list]
    subgraph_labels = y_nrf[subgraph_indices].cpu().numpy()

    train_set, val_set = train_test_split(
        train_data_list,
        test_size=0.2,
        stratify=subgraph_labels,
        random_state=42
    )


    # Step 3: Prepare DataLoaders for subgraph batching
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'])
    test_loader = DataLoader(test_data_list, batch_size=args['batch_size'])

    # Step 4: Loss function and optimizer
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(neural_forest.parameters()), lr=args['lr'])

    y_gcn = y_gcn.to(device)
    y_nrf = y_nrf.to(device)
    nrf_input = nrf_input.to(device)

    best_acc = -1
    best_epoch = -1
    epoch_logs = []
    no_improvement = 0
    patience = 500
    best_state_dict = None

    if verbose:
        #epoch_iter = tqdm(range(args['epochs']), desc="Training epochs")
        epoch_iter = range(args['epochs'])
    else:
        epoch_iter = range(args['epochs'])

    for epoch in epoch_iter:
        start_time = time.time()
        model.train()
        neural_forest.train()
        total_loss = 0

        # Step 5: Train on batched subgraphs
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            gcn_pred = model(batch)  # shape: (batch_size,)
            batch_indices = batch.idx
            gcn_targets = y_gcn[batch_indices]
            nrf_targets = y_nrf[batch_indices]
            other_feats = nrf_input[batch_indices]

            pred_error = (gcn_pred - gcn_targets).pow(2).unsqueeze(1)
            input_features = torch.cat([other_feats, pred_error], dim=1)

            out = neural_forest(input_features)
            loss1 = mse_loss(gcn_pred, gcn_targets)
            loss2 = neural_forest.loss(out, nrf_targets)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Step 6: Evaluate on test set
        model.eval()
        neural_forest.eval()

        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                gcn_pred = model(batch)
                batch_indices = batch.idx
                gcn_targets = y_gcn[batch_indices]
                nrf_targets = y_nrf[batch_indices]
                other_feats = nrf_input[batch_indices]

                pred_error = (gcn_pred - gcn_targets).pow(2).unsqueeze(1)
                input_features = torch.cat([other_feats, pred_error], dim=1)

                out = neural_forest(input_features)
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1)

                all_preds.append(preds)
                all_probs.append(probs)
                all_targets.append(nrf_targets)

        # Step 7: Aggregate predictions and compute accuracy
        y_pred_tensor = torch.cat(all_preds)
        y_proba = torch.cat(all_probs)
        y_true_tensor = torch.cat(all_targets)
        acc = (y_pred_tensor == y_true_tensor).float().mean().item()

        # Step 8: Track best model and compute metrics
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

            y_pred = best_labels.cpu().numpy()
            y_proba_np = best_proba.cpu().numpy()
            y_true = y_true_tensor.cpu().numpy()

            y_pred_decoded = [index_to_label[i] for i in y_pred]
            y_true_decoded = [index_to_label[i] for i in y_true]

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

            roc_auc_weighted = roc_auc_score(y_true, y_proba_np, multi_class='ovr', average='weighted')
            roc_auc_micro = roc_auc_score(y_true, y_proba_np, multi_class='ovr', average='micro')
            roc_auc_macro = roc_auc_score(y_true, y_proba_np, multi_class='ovr', average='macro')

        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if verbose:
            if epoch % 50 == 0 or epoch == 1:
                print(f"Epoch {epoch+1:02d} | Joint Loss: {total_loss/len(train_loader):.4f} | NRF Acc: {acc:.4f}")

        epoch_logs.append(time.time() - start_time)

    print(f"Best acc/epoch: {best_acc:.4f} at epoch {best_epoch}")

    if args['final_evaluation']:
        print("Predicting on test set")
        model.load_state_dict(best_state_dict['gcn'])
        neural_forest.load_state_dict(best_state_dict['ndf'])
        all_preds = []
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                gcn_pred = model(batch)
                batch_indices = batch.idx
                gcn_targets = y_gcn[batch_indices]
                nrf_targets = y_nrf[batch_indices]
                other_feats = nrf_input[batch_indices]

                pred_error = (gcn_pred - gcn_targets).pow(2).unsqueeze(1)
                input_features = torch.cat([other_feats, pred_error], dim=1)

                out = neural_forest(input_features)
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1)

                all_preds.append(preds)
                all_probs.append(probs)
                all_targets.append(nrf_targets)
        # Concatenate batched results
        y_pred_tensor = torch.cat(all_preds)
        y_proba = torch.cat(all_probs)
        y_true_tensor = torch.cat(all_targets)

        # Compute accuracy
        test_acc = (y_pred_tensor == y_true_tensor).float().mean().item()

    if not args["final_evaluation"]:
        test_acc = best_acc

    return (
        test_acc,
        best_epoch,
        precision, recall, f1,
        y_pred_decoded, y_true_decoded,
        precision_micro, recall_micro, f1_micro,
        precision_macro, recall_macro, f1_macro,
        roc_auc_weighted, roc_auc_micro, roc_auc_macro,
        epoch_logs
    )
