import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from collections import defaultdict
from typing import List

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def clean_spanish_mobility_data(file_path: str, centrality_path: str, save_path: str) -> pd.DataFrame:
    """
    Cleans Spanish mobility Excel file:
    - Normalizes VIAJES
    - Renames provinces to match COVID dataset
    - Removes non-Spanish provinces
    - Filters only cities present in the centrality index
    - Saves cleaned output to CSV

    Args:
        file_path (str): Path to the raw mobility Excel file
        centrality_path (str): Path to centrality CSV with valid province names
        save_path (str): Where to save the final cleaned mobility CSV

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """

    rename_map = {
        'Illes Balears': 'Balears',
        'Valencia': 'Valencia/València',
    }
    invalid_entries = {'FR', 'PT', 'ex'}

    # Load centrality cities
    centrality_df = pd.read_csv(centrality_path)
    valid_cities = set(centrality_df['nomemun'])

    # Load mobility data
    df = pd.read_excel(file_path, sheet_name='Data', skiprows=2)
    df = df.dropna(subset=['COD. PROV. ORIGEN', 'COD. PROV. DESTINO', 'VIAJES'])
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])
    df = df[df['COD. PROV. ORIGEN'] != df['COD. PROV. DESTINO']]
    df['VIAJES'] = pd.to_numeric(df['VIAJES'], errors='coerce')
    df = df.dropna(subset=['VIAJES'])

    # Rename provinces
    df['PROVINCIA ORIGEN'] = df['PROVINCIA ORIGEN'].apply(lambda x: rename_map.get(x, x))
    df['PROVINCIA DESTINO'] = df['PROVINCIA DESTINO'].apply(lambda x: rename_map.get(x, x))

    # Filter invalid + only centrality cities
    df = df[
        (~df['PROVINCIA ORIGEN'].isin(invalid_entries)) &
        (~df['PROVINCIA DESTINO'].isin(invalid_entries)) &
        (df['PROVINCIA ORIGEN'].isin(valid_cities)) &
        (df['PROVINCIA DESTINO'].isin(valid_cities))
    ]

    # Normalize VIAJES
    scaler = MinMaxScaler()
    df['VIAJES_NORM'] = scaler.fit_transform(df[['VIAJES']])

    # Rename for output
    df = df.rename(columns={
        'COD. PROV. ORIGEN': 'origin',
        'COD. PROV. DESTINO': 'destination',
        'VIAJES_NORM': 'weight'
    })

    cleaned_df = df[['origin', 'destination', 'weight', 'PROVINCIA ORIGEN', 'PROVINCIA DESTINO']].copy()

    # Save cleaned result
    cleaned_df.to_csv(save_path, index=False)
    print(f"[✓] Final cleaned mobility data saved to: {save_path}")

    return cleaned_df



def build_pyg_graph_from_backbone(
    centrality_df: pd.DataFrame,
    backbone_df: pd.DataFrame
) -> Data:
    """
    Builds a PyTorch Geometric Data object using province names as nodes.

    Args:
        centrality_df (pd.DataFrame): DataFrame with 'nomemun' column (province names).
        backbone_df (pd.DataFrame): DataFrame with 'source', 'target', 'weight' using province names.

    Returns:
        Data: PyTorch Geometric graph object.
    """
    G = nx.Graph()

    # Add all named provinces from centrality_df
    for _, row in centrality_df.iterrows():
        name = row['nomemun']
        G.add_node(name)

    # Add backbone edges (only if both names exist in centrality)
    valid_names = set(centrality_df['nomemun'])
    for _, row in backbone_df.iterrows():
        src, tgt = row['source'], row['target']
        if src in valid_names and tgt in valid_names and src != tgt and not G.has_edge(src, tgt):
            G.add_edge(src, tgt, weight=row['weight'])

    print(f"[✓] Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    for _, _, d in G.edges(data=True):
        d['edge_weight'] = d['weight']

    pyg_data = from_networkx(G, group_edge_attrs=['edge_weight'])
    print("[✓] Converted to PyTorch Geometric format.")
    return pyg_data



def extract_backbone_from_avg_matrix(avg_matrix: pd.DataFrame, cities: list, alpha=0.01, top_k=5) -> pd.DataFrame:
    """
    Extracts a backbone from an average mobility matrix using pij filtering and top-k neighbors.

    Args:
        avg_matrix (pd.DataFrame): Averaged mobility matrix with city names as index/columns.
        cities (list): List of province names (str).
        alpha (float): pij threshold.
        top_k (int): Number of top neighbors to retain.

    Returns:
        pd.DataFrame: Backbone edge list using province names.
    """
    print(f"[✓] Extracting backbone using pij < {alpha} and top-{top_k} neighbors")
    degree = defaultdict(int)
    strength = defaultdict(float)
    edges = []

    for i, src in enumerate(cities):
        for j, dst in enumerate(cities):
            if i != j:
                w = avg_matrix.iloc[i, j]
                if not np.isnan(w) and w > 0:
                    edges.append((src, dst, w))
                    degree[src] += 1
                    strength[src] += w

    records = []
    for src, dst, w in edges:
        si, ki = strength[src], degree[src]
        sj, kj = strength[dst], degree[dst]
        pij_i = (1 - w / si) ** (ki - 1) if ki > 1 and si > 0 else 1
        pij_j = (1 - w / sj) ** (kj - 1) if kj > 1 and sj > 0 else 1
        pij = min(pij_i, pij_j)
        records.append({
            "source": src,
            "target": dst,
            "weight": w,
            "pij": pij
        })

    df_edges = pd.DataFrame(records)
    df_edges["edge_key"] = list(zip(df_edges["source"], df_edges["target"]))

    topk_neighbors = defaultdict(set)
    for node in cities:
        node_edges = df_edges[df_edges["source"] == node]
        topk = node_edges.nlargest(top_k, "weight")
        topk_neighbors[node].update(zip(topk["source"], topk["target"]))

    df_edges["keep_topk"] = df_edges["edge_key"].apply(
        lambda x: x in topk_neighbors[x[0]]
    )

    df_edges["keep"] = (df_edges["pij"] < alpha) | df_edges["keep_topk"]
    df_backbone = df_edges[df_edges["keep"]].copy()
    print(f"[✓] Filtered down to {len(df_backbone)} edges from {len(df_edges)}")

    return df_backbone


def mock_extract_backbone_from_avg_matrix(avg_matrix: pd.DataFrame, cities: List[str], alpha=0.01,
                                          top_k=5) -> pd.DataFrame:
    """
    Mock version of extract_backbone_from_avg_matrix that just returns all non-zero entries as edges.

    Args:
        avg_matrix (pd.DataFrame): Averaged mobility matrix with city names as index/columns.
        cities (List[str]): List of province names (str).
        alpha (float): Ignored.
        top_k (int): Ignored.

    Returns:
        pd.DataFrame: Unfiltered edge list using province names.
    """
    print("[⚙] Mock backbone extractor: returning all non-zero mobility flows.")

    edges = []
    for i, src in enumerate(cities):
        for j, dst in enumerate(cities):
            if i != j:
                w = avg_matrix.iloc[i, j]
                if not pd.isna(w) and w > 0:
                    edges.append({
                        "source": src,
                        "target": dst,
                        "weight": w,
                        "pij": 1.0,  # placeholder
                        "keep_topk": True,
                        "keep": True
                    })

    return pd.DataFrame(edges)

