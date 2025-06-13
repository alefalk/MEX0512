# graph_utils.py
import numpy as np
import pandas as pd

def create_unique_geo_data(df):
    # Convert to datetime and sort
    df['attack_date'] = pd.to_datetime({'year': df['iyear'], 'month': df['imonth'], 'day': df['iday']})
    df.sort_values(by=['longitude', 'latitude', 'attack_date'], inplace=True)

    # Keep only relevant columns
    df = df[['longitude', 'latitude', 'gname']]

    # Drop duplicates based on location, keep the earliest attack
    df_unique = df.drop_duplicates(subset=['longitude', 'latitude'], keep='first').reset_index(drop=True)

    return df_unique


import numpy as np

def build_graph_data(df, coord_to_index):
    """
    Constructs the global adjacency matrix and related data using a fixed node index mapping.

    Args:
        df: DataFrame with 'longlat' and 'gname' columns
        coord_to_index: Dict mapping (longitude, latitude) tuples -> node index, built from full dataset

    Returns:
        - adjacency_matrix: N x N adjacency matrix with self-loops on active nodes
        - feature_matrix: N x N identity matrix (one-hot node features)
        - label_index: Dict mapping group_name -> class index
    """
    N = len(coord_to_index)
    print(f"Number of total nodes (unique coordinates): {N}")

    # Identity feature matrix: one-hot features for each node
    feature_matrix = np.eye(N, dtype=np.float32)
    adjacency_matrix = np.zeros((N, N), dtype=np.float32)

    # Self-loops for active nodes
    for coord in df['combination']:
        idx = coord_to_index[coord]
        adjacency_matrix[idx, idx] = 1.0

    # Label mapping
    unique_labels = df['gname'].unique()
    label_index = {label: i for i, label in enumerate(unique_labels)}
    print(f"Number of unique labels in this set: {len(label_index)}")

    return adjacency_matrix, feature_matrix, label_index




