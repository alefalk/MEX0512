# Replace RandomForestClassifier with NeuralDecisionForest in train_joint
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from ndf_old import UCIAdultFeatureLayer, Forest, NeuralDecisionForest

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
    # ?
    full_coords = df['longlat'].drop_duplicates().reset_index(drop=True)
    # Map coordinates to unique node indices
    coord_to_index = {coord: i for i, coord in enumerate(full_coords)}
    # Number of nodes, N
    N = len(coord_to_index)

    # identity matrix, self loops only
    adj = np.eye(N)
    # 2 x N tensor with edges from each node to itself
    edge_index = torch.tensor(np.vstack(coo_matrix(adj).nonzero()), dtype=torch.long)
    # Feature matrix containing [longitude, latitude] for each node, shape N x 2
    x = torch.tensor(np.array(list(coord_to_index.keys())), dtype=torch.float32)

    # Initialize labels and masks
    y_gcn = torch.full((N,), -1.0, dtype=torch.float32) # -1 marks unknown/unlabeled nodes
    y_nrf = np.full((N,), -1, dtype=int)
    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    # split data, handle leakage
    train_df, test_df = handle_leakage(df)

    for _, row in train_df.iterrows():
        coord = row['longlat'] # Find coordinate
        if coord in coord_to_index:
            idx = coord_to_index[coord]            # Get node index 
            y_gcn[idx] = row[continuous_col]       # Set target, continuous value
            y_nrf[idx] = label_index[row['gname']] # Set classification label
            train_mask[idx] = True                 # Update mask

    # Same as above
    for _, row in test_df.iterrows():
        coord = row['longlat']
        if coord in coord_to_index:
            idx = coord_to_index[coord]
            y_gcn[idx] = row[continuous_col]
            y_nrf[idx] = label_index[row['gname']]
            test_mask[idx] = True


    # Define the other features to be fed into the nrf
    non_location_cols = [col for col in df.columns if col not in ['longlat', 'gname', continuous_col]]
    non_geo_features = torch.zeros((N, len(non_location_cols)))


    for _, row in df.iterrows():
        coord = (row['longlat'])
        if coord in coord_to_index:
            idx = coord_to_index[coord] # Get id for attack
            row_values = row[non_location_cols].astype(float).values # get features from that attack
            non_geo_features[idx] = torch.tensor(row_values, dtype=torch.float32) # Build non_geo_features tensor

    #df = pd.get_dummies(df, columns=['longlat'])


    return Data(x=x, edge_index=edge_index), y_gcn, y_nrf, non_geo_features, train_mask, test_mask

def handle_leakage(df):
    train_frames = []
    test_frames = []
    for _, group in df.groupby('gname'):
        split = int(len(group) * 0.7)
        train_frames.append(group.iloc[:split])
        test_frames.append(group.iloc[split:])
    return shuffle(pd.concat(train_frames)), shuffle(pd.concat(test_frames))

def train_joint(data, edge_index, y_gcn, y_nrf, non_geo_features, train_mask, test_mask, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNRegressor(data.num_node_features, args['embed_dim']).to(device)
    feat_layer = UCIAdultFeatureLayer(dropout_rate=args['feat_dropout'])
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

    data = data.to(device)
    y_gcn = y_gcn.to(device)
    non_geo_features = non_geo_features.to(device)
    y_nrf = torch.tensor(y_nrf, dtype=torch.long).to(device)

    best_acc = -1
    best_epoch = -1

    for epoch in range(args['epochs']):
        #Train GCN and NRF
        model.train()
        neural_forest.train()

        #Zero gradients
        optimizer.zero_grad()

        #Predict GCN and calculate error
        pred = model(data.x, edge_index.to(device))
        pred_error = (pred - y_gcn).pow(2).unsqueeze(1)
        
        #Input to NRF
        input_features = torch.cat([non_geo_features, pred_error], dim=1)

        # Compute both losses
        loss1 = mse_loss(pred[train_mask], y_gcn[train_mask])
        out_forest = neural_forest(input_features)
        loss2 = neural_forest.loss(out_forest[train_mask], y_nrf[train_mask])
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        # Evaluate NRF
        model.eval()
        neural_forest.eval()
        with torch.no_grad():
            pred_eval = model(data.x, edge_index.to(device))
            pred_error = (pred_eval - y_gcn).pow(2).unsqueeze(1)
            input_features = torch.cat([non_geo_features, pred_error], dim=1)
            out_forest = neural_forest(input_features)
            pred_labels = out_forest[test_mask].argmax(dim=1)
            acc = (pred_labels == y_nrf[test_mask]).float().mean().item()

            if acc>best_acc:
                best_acc = acc
                best_epoch = epoch

        print(f"Epoch {epoch+1:02d} | GCN MSE Loss: {loss1.item():.4f} | NRF Loss: {loss2.item():.4f} | JOINT Loss: {loss.item():.4f} | NRF Acc: {acc:.4f}")

    print(f"Best acc/epoch: {best_acc}, epoch {best_epoch}")