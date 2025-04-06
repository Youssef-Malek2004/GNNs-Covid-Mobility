import numpy as np
import pandas as pd
from collections import defaultdict

def extract_backbone_from_avg_matrix(avg_matrix, cities, alpha=0.01, top_k=5):
    """
    Extract backbone from average mobility matrix using pij filtering and top-k neighbor rule.

    Args:
        avg_matrix (pd.DataFrame): Averaged mobility matrix (rows and columns should be same city list).
        cities (list): List of city names corresponding to rows/columns in avg_matrix.
        alpha (float): pij threshold for filtering.
        top_k (int): Number of top neighbors to preserve per node.

    Returns:
        pd.DataFrame: Backbone edge list with 'source', 'target', 'weight', 'pij', 'keep_topk', 'keep'
    """
    print(f"[✓] Extracting backbone using pij < {alpha} and top-{top_k} neighbors")

    degree = defaultdict(int)
    strength = defaultdict(float)
    edges = []

    # Collect edges
    for i, src in enumerate(cities):
        for j, dst in enumerate(cities):
            if i != j:
                w = avg_matrix.iloc[i, j]
                if not np.isnan(w) and w > 0:
                    edges.append((i, j, w))
                    degree[i] += 1
                    strength[i] += w

    # Calculate pij
    records = []
    for i, j, w in edges:
        si, ki = strength[i], degree[i]
        sj, kj = strength[j], degree[j]

        pij_i = (1 - w / si) ** (ki - 1) if ki > 1 and si > 0 else 1
        pij_j = (1 - w / sj) ** (kj - 1) if kj > 1 and sj > 0 else 1
        pij = min(pij_i, pij_j)

        records.append({
            "source": cities[i],
            "target": cities[j],
            "weight": w,
            "pij": pij
        })

    df_edges = pd.DataFrame(records)
    df_edges["edge_key"] = list(zip(df_edges["source"], df_edges["target"]))

    # Top-k strongest neighbors
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
