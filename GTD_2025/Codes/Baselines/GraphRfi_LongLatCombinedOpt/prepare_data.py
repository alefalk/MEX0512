import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from haversine import haversine, Unit

# ====== Subgraph Extraction Function ======
def get_subgraph(center_id, edge_index, x_global):
    neighbors = edge_index[1][edge_index[0] == center_id]
    node_ids = torch.cat([torch.tensor([center_id]), neighbors]).unique()

    id_map = {old_id.item(): i for i, old_id in enumerate(node_ids)}
    new_edges = []
    for source, destination in zip(*edge_index):
        if source in node_ids and destination in node_ids:
            new_edges.append((id_map[source.item()], id_map[destination.item()]))

    if len(new_edges) == 0:
        center_local_idx = 0
        new_edges = [(0, 0)]
    else:
        center_local_idx = id_map[center_id.item()]

    sub_x = x_global[node_ids]
    sub_edge_index = torch.tensor(new_edges).T

    return sub_x, sub_edge_index, center_local_idx

# ====== Data Object Construction ======
def build_data_object(row, location2id, global_edge_index, x_global, label_encoder, ndf_feature_columns):
    loc = (row['longitude'], row['latitude'])
    if loc not in location2id:
        return None

    center_id = location2id[loc]
    label = label_encoder.transform([row['gname']])[0]

    x, edge_index, center_idx = get_subgraph(
        torch.tensor(center_id),
        global_edge_index,
        torch.tensor(x_global, dtype=torch.float)
    )

    weaptype1 = torch.tensor(row['weaptype1'], dtype=torch.float)
    other_features = torch.tensor(row[ndf_feature_columns].values.astype(np.float32), dtype=torch.float).unsqueeze(0)

    return Data(
        x=x,
        edge_index=edge_index,
        center=center_idx,
        y=torch.tensor(label),
        weaptype1=weaptype1,
        ndf_features=other_features
    )

# ====== Main Data Prep Function ======
def prepare_data(train_csv, test_csv):
    # Load CSVs
    traindata = pd.read_csv(train_csv, encoding='ISO-8859-1')
    testdata = pd.read_csv(test_csv, encoding='ISO-8859-1')
    combined = pd.concat([traindata, testdata], axis=0)

    # Create location identifiers
    combined['location'] = list(zip(combined['longitude'], combined['latitude']))
    unique_locations = combined['location'].drop_duplicates().reset_index(drop=True)
    location2id = {loc: idx for idx, loc in enumerate(unique_locations)}
    combined['location_id'] = combined['location'].map(location2id)

    # Encode gname labels
    le = LabelEncoder()
    combined['label'] = le.fit_transform(combined['gname'])

    # Extract coordinates and standardize
    coords = np.array([list(loc) for loc in unique_locations])
    scaler = StandardScaler()
    x_global = scaler.fit_transform(coords)

    # Haversine edges
    edges = []
    coords_latlon = [(lat, lon) for lon, lat in unique_locations]
    for i in range(len(coords_latlon)):
        for j in range(i + 1, len(coords_latlon)):
            if haversine(coords_latlon[i], coords_latlon[j], Unit.KILOMETERS) <= 1.0:
                edges.append((i, j))
                edges.append((j, i))
    global_edge_index = torch.tensor(edges, dtype=torch.long).T

    # Columns to exclude from NDF features
    exclude = {'gname', 'location', 'location_id', 'label'}
    all_columns = set(traindata.columns)
    ndf_feature_columns = sorted(list(all_columns - exclude))

    # Construct PyG Data objects
    all_data = []
    for _, row in traindata.iterrows():
        data_obj = build_data_object(row, location2id, global_edge_index, x_global, le, ndf_feature_columns)
        if data_obj is not None:
            all_data.append(data_obj)

    # Train/val split
    train_list, val_list = train_test_split(all_data, test_size=0.2, random_state=42)

    return train_list, val_list, le.classes_

# ====== Example Usage ======
if __name__ == "__main__":
    train_path = "../../../data/top30groups/LongLatCombined/train1/train100.csv"
    test_path = "../../../data/top30groups/LongLatCombined/test1/test100.csv"

    train_list, val_list, class_names = prepare_data(train_path, test_path)
    print(f"Train size: {len(train_list)}, Val size: {len(val_list)}, Classes: {len(class_names)}")
