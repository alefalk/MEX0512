# Standard PyTorch + math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Custom Neural Decision Forest (adapted from GraphRfi's ndf.py)
import graphrfi_gtd  # This assumes ndf.py is in the same directory

# Optional: Utilities
import numpy as np
import pandas as pd

class GraphRfi(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        return x

    def get_embedding(self, x, edge_index, nodes=None):
        x = self.forward(x, edge_index)
        return x if nodes is None else x[nodes]
    

def build_graph_data(df, label_index):
    # 1. Create a unique set of coordinates and assign node indices
    coords_df = df[['longitude', 'latitude']].drop_duplicates().reset_index(drop=True)
    coord_to_index = {(row['longitude'], row['latitude']): i for i, row in coords_df.iterrows()}
    N = len(coord_to_index)

    # 2. Build the adjacency matrix (self-loop only)
    adj = np.eye(N, dtype=np.float32)
    A_coo = coo_matrix(adj)
    edge_index = torch.tensor(np.vstack((A_coo.row, A_coo.col)), dtype=torch.long)

    # 3. Feature matrix = raw coordinates
    x = torch.tensor(coords_df.values, dtype=torch.float32)  # shape [N, 2]

    # 4. Label vector
    y = torch.full((N,), -1, dtype=torch.long)

    # 5. Split train/test using your group-wise strategy
    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    train_coords = []
    test_coords = []

    for _, group in df.groupby('gname'):
        group = group.drop_duplicates(subset=['longitude', 'latitude'])
        split = int(len(group) * 0.7)
        train_coords += list(group.iloc[:split][['longitude', 'latitude']].itertuples(index=False, name=None))
        test_coords += list(group.iloc[split:][['longitude', 'latitude']].itertuples(index=False, name=None))

    # 6. Assign labels and masks
    for coord in train_coords:
        if coord in coord_to_index:
            idx = coord_to_index[coord]
            gname = df[(df['longitude'] == coord[0]) & (df['latitude'] == coord[1])]['gname'].iloc[0]
            y[idx] = label_index[gname]
            train_mask[idx] = True

    for coord in test_coords:
        if coord in coord_to_index:
            idx = coord_to_index[coord]
            gname = df[(df['longitude'] == coord[0]) & (df['latitude'] == coord[1])]['gname'].iloc[0]
            y[idx] = label_index[gname]
            test_mask[idx] = True

    data = Data(x=x, edge_index=edge_index)
    return data, y, train_mask, test_mask


def train(model, data, y, train_mask, neuralforest, optimizer):
    model.train()
    neuralforest.train()

    x, edge_index = data.x, data.edge_index
    embeddings = model(x, edge_index)
    x_train = embeddings[train_mask]
    y_train = y[train_mask]

    logits = neuralforest(x_train)
    loss = neuralforest.loss(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()



def test(model, data, y, test_mask, neuralforest):
    model.eval()
    neuralforest.eval()

    x, edge_index = data.x, data.edge_index
    with torch.no_grad():
        embeddings = model(x, edge_index)
        x_test = embeddings[test_mask]
        y_test = y[test_mask]

        logits = neuralforest(x_test)
        pred = logits.argmax(dim=1)
        acc = (pred == y_test).float().mean().item()

    return acc


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--feat_dropout', type=float, default=0.3)
    parser.add_argument('--n_tree', type=int, default=80)
    parser.add_argument('--tree_depth', type=int, default=10)
    parser.add_argument('--tree_feature_rate', type=float, default=0.5)
    args = parser.parse_args(args=[])  # for notebook; remove `args=[]` for script

    # Load data
    df = pd.read_csv("top30groups/df_top30_100.csv")  # Replace with actual path
    unique_labels = sorted(df['gname'].unique())
    label_index = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(label_index)

    # Prepare graph data and masks
    data, y, train_mask, test_mask = build_graph_data(df, label_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)

    # Initialize models
    model = GraphRfi(in_channels=data.num_node_features, hidden_channels=args.embed_dim).to(device)
    feat_layer = ndf.UCIAdultFeatureLayer(dropout_rate=args.feat_dropout)
    forest = ndf.Forest(
        n_tree=args.n_tree,
        tree_depth=args.tree_depth,
        n_in_feature=feat_layer.get_out_feature_size(),
        tree_feature_rate=args.tree_feature_rate,
        n_class=num_classes,
        jointly_training=True
    )
    neuralforest = ndf.NeuralDecisionForest(feat_layer, forest).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(neuralforest.parameters()), lr=args.lr
    )

    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        loss = train(model, data, y, train_mask, neuralforest, optimizer)
        acc = test(model, data, y, test_mask, neuralforest)
        best_acc = max(best_acc, acc)

        print(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")

    print(f"\n✅ Best Test Accuracy: {best_acc:.4f}")

