# graph_utils.py
import numpy as np
import pandas as pd

def create_data(df):
    df['attack_date'] = pd.to_datetime({'year': df['iyear'], 'month': df['imonth'], 'day': df['iday']})
    df.sort_values(by=['gname', 'attack_date'], inplace=True)
    df_unique = df.drop_duplicates(subset='gname', keep='first')
    df_unique = df_unique[['longitude', 'latitude', 'gname']]
    print(df_unique)
    return df_unique

def build_graph_data(df_unique, coord_to_index=None):
    """
    Given a dataframe with columns ['longitude', 'latitude', 'gname'],
    return:
        - adjacency_matrices: dict of {group_name: adjacency_matrix}
        - node_indices: dict of {group_name: list of node indices used in the graph}
        - coord_to_index: dict mapping (longitude, latitude) to node index
        - feature_matrix: identity matrix (N x N)
        - label_index: mapping from group name to class index
    """
    # If no coord_to_index is provided, build it from this dataset
    if coord_to_index is None:
        coordinates_unique = df_unique[['longitude', 'latitude']].drop_duplicates().reset_index(drop=True)
        coordinate_pairs = coordinates_unique.to_records(index=False).tolist()
        coord_to_index = {coord: i for i, coord in enumerate(coordinate_pairs)}

    N = len(coord_to_index)
    print(f"Number of unique coordinates (nodes): {N}")

    # Feature matrix: one-hot encoding for each node (identity matrix)
    feature_matrix = np.eye(N, dtype='float32')

    # Encode labels
    unique_labels = df_unique['gname'].unique()
    print("Number of unique labels: ", len(unique_labels))
    label_index = {label: i for i, label in enumerate(unique_labels)}

    # Build adjacency matrices and track active node indices per group
    adjacency_matrices = {}
    group_node_indices = {}

    grouped = df_unique.groupby('gname')
    for group_name, group_df in grouped:
        coords = list(zip(group_df['longitude'], group_df['latitude']))
        node_indices = [coord_to_index[coord] for coord in coords if coord in coord_to_index]

        A = np.zeros((N, N), dtype=np.float32)
        for idx in node_indices:
            A[idx, idx] = 1.0  # Self-loop

        adjacency_matrices[group_name] = A
        group_node_indices[group_name] = node_indices

    return adjacency_matrices, group_node_indices, coord_to_index, feature_matrix, label_index

