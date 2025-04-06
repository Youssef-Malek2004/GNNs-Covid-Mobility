import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import pandas as pd


def build_pyg_graph_from_backbone(
    centrality_df: pd.DataFrame,
    backbone_df: pd.DataFrame
) -> Data:
    """
    Builds a PyTorch Geometric Data object from centrality + backbone edges.

    Args:
        centrality_df (pd.DataFrame): DataFrame with at least 'Codmundv' (int) and 'nomemun' columns.
        backbone_df (pd.DataFrame): DataFrame with 'source', 'target', 'weekly_flow'.

    Returns:
        Data: PyTorch Geometric graph object with edge weights.
    """

    # 1. Create undirected weighted NetworkX graph
    G = nx.Graph()

    # 2. Add nodes
    for _, row in centrality_df.iterrows():
        city_id = int(row['Codmundv'])
        G.add_node(city_id, name=row['nomemun'])  # 'name' not used by PyG unless manually extracted

    # 3. Add backbone edges
    for _, row in backbone_df.iterrows():
        source, target = int(row['source']), int(row['target'])
        if source != target and not G.has_edge(source, target):
            G.add_edge(source, target, weight=row['weekly_flow'])

    print(f"[✓] Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # 4. Add edge_weight for PyG conversion
    for _, _, d in G.edges(data=True):
        d['edge_weight'] = d['weight']

    # 5. Convert to PyTorch Geometric
    pyg_data = from_networkx(G, group_edge_attrs=['edge_weight'])

    print("[✓] Converted to PyTorch Geometric format.")
    return pyg_data
