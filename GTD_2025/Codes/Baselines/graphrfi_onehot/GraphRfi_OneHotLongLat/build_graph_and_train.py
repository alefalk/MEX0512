import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from ndf import GTD100FeatureLayer, GTD200FeatureLayer, GTD300FeatureLayer, GTD478FeatureLayer, Forest, NeuralDecisionForest

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

    # Step 3: NRF input: one-hot encode longlat (full df)
    df_onehot = pd.get_dummies(df['longlat'])
    longlat_onehot = torch.tensor(df_onehot.values, dtype=torch.float32)  # shape: (len(df), N)

    # Step 4: Build non-location features (full df)
    non_location_cols = [col for col in df.columns if col not in ['longlat', 'gname']]
    other_feats = torch.tensor(df[non_location_cols].astype(float).values, dtype=torch.float32)

    # Step 5: Concatenate one-hot + other features for NRF input
    nrf_input = torch.cat([longlat_onehot, other_feats], dim=1)

    # Step 6: Map each row to a node
    row_to_node_index = torch.tensor(df['longlat'].map(coord_to_index).values, dtype=torch.long)

    # Step 7: Create NRF labels (one per row)
    y_nrf = torch.tensor(df['gname'].map(label_index).values, dtype=torch.long)

    return Data(x=x, edge_index=edge_index), y_gcn, y_nrf, nrf_input, train_mask, test_mask, row_to_node_index



def handle_leakage(df):
    train_frames = []
    test_frames = []
    for _, group in df.groupby('gname'):
        split = int(len(group) * 0.7)
        train_frames.append(group.iloc[:split])
        test_frames.append(group.iloc[split:])
    return shuffle(pd.concat(train_frames)), shuffle(pd.concat(test_frames))

def train_joint(data, edge_index, y_gcn, y_nrf, non_geo_features, train_mask, test_mask, args, row_to_node_index):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNRegressor(data.num_node_features, args['embed_dim']).to(device)

    # Depending on what partition
    if args['partition'] == "gtd100":
        feat_layer = GTD100FeatureLayer(dropout_rate=args['feat_dropout'])
    elif args['partition'] == "gtd200":
        print("hej")
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
    optimizer = torch.optim.Adam(list(model.parameters()) + list(neural_forest.parameters()), lr=args['lr'])
    mse_loss = torch.nn.MSELoss()

    # GPU Compatible
    data = data.to(device)
    y_gcn = y_gcn.to(device)
    non_geo_features = non_geo_features.to(device)
    y_nrf = torch.tensor(y_nrf, dtype=torch.long).to(device)

    best_acc = -1
    best_epoch = -1

    row_train_mask = train_mask[row_to_node_index]
    row_test_mask = test_mask[row_to_node_index]
    for epoch in range(args['epochs']):
        #Train GCN and NRF
        model.train()
        neural_forest.train()

        #Zero gradients
        optimizer.zero_grad()

        #Predict GCN and calculate error
        pred = model(data.x, edge_index.to(device))
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
            pred_eval = model(data.x, edge_index.to(device))
            pred_error = (pred_eval - y_gcn).pow(2).unsqueeze(1)
            per_row_pred_error = pred_error[row_to_node_index]
            input_features = torch.cat([non_geo_features, per_row_pred_error], dim=1)
            out_forest = neural_forest(input_features)
            pred_labels = out_forest[row_test_mask].argmax(dim=1)
            acc = (pred_labels == y_nrf[row_test_mask]).float().mean().item()

            if acc>best_acc:
                best_acc = acc
                best_epoch = epoch

        print(f"Epoch {epoch+1:02d} | GCN MSE Loss: {loss1.item():.4f} | NRF Loss: {loss2.item():.4f} | JOINT Loss: {loss.item():.4f} | NRF Acc: {acc:.4f}")

    print(f"Best acc/epoch: {best_acc}, epoch {best_epoch}")
    return best_acc, best_epoch