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



class GCNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)  # Output size should be num_classes

    def forward(self, x, edge_index):
        # First convolution layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Return embeddings (after first conv layer)
        embeddings = x

        # Second convolution layer (logits for classification)
        x = self.conv2(x, edge_index)
        
        return x, embeddings  # Return both logits (for classification) and embeddings


def haversine_edges(coords, threshold_km=1.0):

    edges = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if haversine(coords[i], coords[j], Unit.KILOMETERS) <= threshold_km:
                edges.append((i, j))
                edges.append((j, i))
    return edges

""" 
If your goal is transductive learning
(Like in Cora, Citeseer, Pubmed GCN benchmarks)

Nodes are fixed; the split is over nodes in a single graph.

You expect the same node to appear in train/val/test, because the graph is one entity.

Leakage risk is only in labels: as long as you don’t feed test labels into training, you’re fine.

In this setting, if a coordinate (node) appears in train and test rows, that’s not leakage 
— it’s the point of transductive training. You’re just evaluating whether the model 
learned good embeddings for known nodes.
"""

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
    y_gcn = torch.full((N,), -1.0, dtype=torch.long)
    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)

    train_df, val_df, test_df = handle_leakage(df)

    def mark(df_split, mask):
        for _, row in df_split.iterrows():
            coord = row['longlat']
            if coord in coord_to_index:
                idx = coord_to_index[coord]
                gcn_target = label_index.get(row['gname'])
                if gcn_target is None:
                    raise ValueError(f"Unknown label {row['gname']}")
                y_gcn[idx] = gcn_target
                #print(gcn_target)
                mask[idx] = True

    mark(train_df, train_mask)
    mark(val_df, val_mask)
    mark(test_df, test_mask)

    #print(y_gcn.shape)                      # torch.Size([num_nodes])
    #print((y_gcn == -1).sum().item())       # how many unlabeled nodes
    #print(torch.unique(y_gcn[y_gcn!=-1]))

    print(train_mask.sum())
    print(val_mask.sum())
    print(test_mask.sum())

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
    #print(y_nrf.shape)
    #print(y_nrf)
    index_to_label = {v: k for k, v in label_index.items()}

    print(train_mask[row_to_node_index].sum())

    return full_coords, Data(x=x, edge_index=edge_index_train), edge_index_full, y_gcn, y_nrf, train_df, val_df, test_df, train_mask, val_mask, test_mask, row_to_node_index, index_to_label

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

def train_joint(df, full_coords, data, edge_index_full, y_gcn, y_nrf, train_df, val_df, test_df,
                train_mask, val_mask, test_mask, args,
                row_to_node_index, index_to_label, verbose):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Store full edge index on data for convenience
    data.edge_index_full = edge_index_full

    model = GCNClassifier(data.num_node_features, args['embed_dim'], args['n_class']).to(device)

    # Select feature extractor
    input_size = len(df.columns) - 2 + args["embed_dim"]
    feat_layer_cls = {
        "gtd100": GTD100FeatureLayer,
        "gtd200": GTD200FeatureLayer,
        "gtd300": GTD300FeatureLayer,
        "gtd478": GTD478FeatureLayer,
    }[args['partition']]

    feat_layer = feat_layer_cls(input_size = input_size, dropout_rate=args['feat_dropout'])

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

    # Dtypes and device
    data = data.to(device)
    y_gcn = y_gcn.to(device).long()                   # cross_entropy needs Long
    y_nrf = y_nrf.to(device).long()
    #non_geo_features = non_geo_features.to(device)

    # masks in row-space (length = num_rows)
    row_train_mask = train_mask[row_to_node_index].to(device)
    row_val_mask   = val_mask[row_to_node_index].to(device)
    row_test_mask  = test_mask[row_to_node_index].to(device)

    best_acc = -1
    best_epoch = -1
    best_state_dict = None
    epoch_logs = []
    no_improvement = 0

    patience = 300 if args.get('final_evaluation', True) else 100

    for epoch in range(args['epochs']):
        start_time = time.time()

        model.train()
        neural_forest.train()
        optimizer.zero_grad()

                # assume you already decided which columns are non-geo features
        non_location_cols = [c for c in df.columns if c not in ["longlat", "gname"]]

        # Build the full (all-rows) feature tensor once, in df row order
        non_geo_features = torch.tensor(
            df[non_location_cols].astype(float).values, dtype=torch.float32, device=device
        )

        pred_nodes, embeddings = model(data.x, data.edge_index)

        # emb_per_row is already in df row order: embeddings[row_to_node_index]
        emb_per_row = embeddings[row_to_node_index.to(device)]  # [num_rows, embed_dim]

        # === Use train_df directly ===
        train_row_idx = torch.tensor(train_df.index.values, dtype=torch.long, device=device)

        feat_train = non_geo_features.index_select(0, train_row_idx)
        emb_train  = emb_per_row.index_select(0, train_row_idx)

        input_features_train = torch.cat([feat_train, emb_train], dim=1)
        out_forest_train = neural_forest(input_features_train)

        # ---- Forward GCN on current training graph (transductive if your edge_index includes val/test) ----
        #pred_nodes, embeddings = model(data.x, data.edge_index)  # pred_nodes: [num_nodes, n_class], embeddings: [num_nodes, embed_dim]

        # Map node embeddings to rows (vectorized)
        #emb_per_row = embeddings[row_to_node_index.to(device)]   # [num_rows, embed_dim]

        # NRF only sees TRAIN rows
        #input_features_train = torch.cat([non_geo_features[row_train_mask],
         #                                 emb_per_row[row_train_mask]], dim=1)
        #out_forest_train = neural_forest(input_features_train)

        # Losses
        loss1 = F.cross_entropy(pred_nodes[train_mask], y_gcn[train_mask])   # node-level CE
        train_idx = torch.as_tensor(train_df.index.values, dtype=torch.long, device=device)
        loss2 = neural_forest.loss(out_forest_train, y_nrf.index_select(0, train_idx))
        #loss2 = neural_forest.loss(out_forest_train, y_nrf[row_train_mask])  # row-level NRF loss on train rows
        loss  = loss1 + loss2

        loss.backward()
        optimizer.step()

        # ---- Validation ----
        model.eval()
        neural_forest.eval()
        with torch.no_grad():
            val_row_idx = torch.tensor(val_df.index.values, dtype=torch.long, device=device)
            feat_val = non_geo_features.index_select(0, val_row_idx)
            emb_val  = emb_per_row.index_select(0, val_row_idx)
            input_features_val = torch.cat([feat_val, emb_val], dim=1)
            
            out_val = neural_forest(input_features_val)

            pred_labels = out_val.argmax(dim=1)
            
            val_idx = torch.as_tensor(val_df.index.values, dtype=torch.long, device=device)
            y_val_true = y_nrf.index_select(0, val_idx)
            acc = (pred_labels == y_val_true).float().mean().item()

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_labels = pred_labels
                best_proba = F.softmax(out_val, dim=1)
                best_state_dict = {
                    'gcn': model.state_dict(),
                    'ndf': neural_forest.state_dict()
                }
                no_improvement = 0

                # Metrics on validation split
                y_true_tensor = y_nrf.index_select(0, val_idx)
                y_pred_tensor = pred_labels
                y_proba = best_proba.detach().cpu().numpy()
                y_true = y_true_tensor.detach().cpu().numpy()

                best_precision = Precision(task='multiclass', average='weighted', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()
                best_recall    = Recall(   task='multiclass', average='weighted', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()
                best_f1        = F1Score(  task='multiclass', average='weighted', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()

                best_precision_micro = Precision(task='multiclass', average='micro', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()
                best_recall_micro    = Recall(   task='multiclass', average='micro', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()
                best_f1_micro        = F1Score(  task='multiclass', average='micro', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()

                best_precision_macro = Precision(task='multiclass', average='macro', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()
                best_recall_macro    = Recall(   task='multiclass', average='macro', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()
                best_f1_macro        = F1Score(  task='multiclass', average='macro', num_classes=args['n_class']).to(device)(y_pred_tensor, y_true_tensor).item()

                # ROC-AUC (OVR) on validation
                roc_auc_weighted = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                roc_auc_micro    = roc_auc_score(y_true, y_proba, multi_class='ovr', average='micro')
                roc_auc_macro    = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | GCN Loss: {loss1.item():.4f} | NRF Loss: {loss2.item():.4f} | Joint: {loss.item():.4f} | Val Acc: {acc:.4f}")

        epoch_logs.append(time.time() - start_time)

    print(f"Best validation acc: {best_acc:.4f} @ epoch {best_epoch}")

    # -------------------- Final evaluation on test set (semi-transductive/inductive) --------------------
    if args['final_evaluation']:
            print("Evaluating on test set using full graph...")
            model.load_state_dict(best_state_dict['gcn'])
            neural_forest.load_state_dict(best_state_dict['ndf'])

            model.eval()
            neural_forest.eval()

            with torch.no_grad():

                test_row_idx = torch.tensor(test_df.index.values, dtype=torch.long, device=device)

                feat_test = non_geo_features.index_select(0, test_row_idx)
                emb_test  = emb_per_row.index_select(0, test_row_idx)
                input_features_test = torch.cat([feat_test, emb_test], dim=1)

                out_test = neural_forest(input_features_test)     # shape: [#test_rows, n_class]

                pred_labels = out_test.argmax(dim=1)

                test_idx = torch.as_tensor(test_df.index.values, dtype=torch.long, device=device)
                y_test_true = y_nrf.index_select(0, test_idx)
                test_acc = (pred_labels == y_test_true).float().mean().item()
                y_true = y_test_true.detach().cpu().numpy()

                # Don't mask again — out_test is already test-only
                pred_proba = F.softmax(out_test, dim=1)

                y_pred = pred_labels.cpu().numpy()
                #y_proba = pred_proba.cpu().numpy()
                #y_true = y_nrf[row_test_mask].cpu().numpy()

                y_pred_decoded = [index_to_label[i] for i in y_pred]
                y_true_decoded = [index_to_label[i] for i in y_true]
    else:
        test_acc = best_acc
        y_pred_decoded = [index_to_label[i] for i in best_labels.cpu().numpy()]
        y_true_decoded = [index_to_label[i] for i in y_nrf[row_val_mask].cpu().numpy()]

    # Return everything you were already returning (so your file-writing stays intact)
    return (
        test_acc, best_epoch, best_precision, best_recall, best_f1,
        y_pred_decoded, y_true_decoded,
        best_precision_micro, best_recall_micro, best_f1_micro,
        best_precision_macro, best_recall_macro, best_f1_macro,
        roc_auc_weighted, roc_auc_micro, roc_auc_macro,
        epoch_logs
    )

