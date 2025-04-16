import os
import pandas as pd
from collections import defaultdict
from typing import Optional, Set


def extract_backbone_from_files_brazil(
        centrality_path: str,
        mobility_edges_path: str,
        alpha: float = 0.01,
        output_path: str = "data/mobility_backbone_brazil.csv",
        city_whitelist: Optional[Set[int]] = None
):
    """
    Extract backbone from mobility edges based on pij threshold and top-5 neighbor rule.
    If the backbone CSV already exists, load and return it directly.

    Args:
        centrality_path (str): Path to centrality Excel file containing 'Codmundv'.
        mobility_edges_path (str): Path to mobility Excel file with 'CODMUNDV_A', 'CODMUNDV_B', and 'VAR05'.
        alpha (float): Threshold for filtering weak edges using pij formula.
        output_path (str): Path to save or load the backbone CSV.
        city_whitelist (Optional[Set[int]]): List of city white list to exclude.
    Returns:
        pd.DataFrame: Filtered edges (backbone only).
    """

    # Check if the output file already exists
    if os.path.exists(output_path):
        print(f"[✓] Backbone file found at '{output_path}'. Loading it...")
        return pd.read_csv(output_path)

    print("[⚙] Backbone file not found. Extracting now...")

    # Load centrality data
    centrality_df = pd.read_excel(centrality_path)
    centrality_city_ids = set(centrality_df['Codmundv'].dropna().astype(int).unique())
    valid_nodes = centrality_city_ids & city_whitelist if city_whitelist else centrality_city_ids

    # Load and clean edge data
    edges_df = pd.read_excel(mobility_edges_path)
    edges_df.rename(columns={
        'CODMUNDV_A': 'source',
        'CODMUNDV_B': 'target',
        'VAR05': 'weekly_flow'
    }, inplace=True)

    # Keep only valid node pairs
    edges_df = edges_df[
        edges_df['source'].isin(valid_nodes) & edges_df['target'].isin(valid_nodes)
    ].copy()

    # Calculate degree and strength
    degree = defaultdict(int)
    strength = defaultdict(float)

    for _, row in edges_df.iterrows():
        i, j = int(row['source']), int(row['target'])
        w = row['weekly_flow']
        degree[i] += 1
        degree[j] += 1
        strength[i] += w
        strength[j] += w

    # Compute pij values
    pij_list = []
    for _, row in edges_df.iterrows():
        i, j = int(row['source']), int(row['target'])
        Aij = row['weekly_flow']
        si, ki = strength[i], degree[i]
        sj, kj = strength[j], degree[j]

        pij_i = (1 - Aij / si) ** (ki - 1) if ki > 1 and si > 0 else 1
        pij_j = (1 - Aij / sj) ** (kj - 1) if kj > 1 and sj > 0 else 1
        pij_list.append(min(pij_i, pij_j))

    edges_df['pij'] = pij_list
    edges_df['edge_key'] = list(zip(edges_df['source'], edges_df['target']))

    # Top 5 edges per node
    top5_neighbors = defaultdict(set)
    for node in valid_nodes:
        neighbors = edges_df[
            (edges_df['source'] == node) | (edges_df['target'] == node)
        ].copy()
        neighbors['weight'] = neighbors['weekly_flow']
        neighbors = neighbors.sort_values(by='weight', ascending=False).head(5)
        for _, r in neighbors.iterrows():
            top5_neighbors[node].add((r['source'], r['target']))

    edges_df['keep_top5'] = edges_df['edge_key'].apply(
        lambda x: x in top5_neighbors[x[0]] or x in top5_neighbors[x[1]]
    )

    # Final keep flag
    edges_df['keep'] = (edges_df['pij'] < alpha) | (edges_df['keep_top5'])

    # Filtered backbone
    backbone_df = edges_df[edges_df['keep']].copy()

    # Save it for future use
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    backbone_df.to_csv(output_path, index=False)
    print(f"[✓] Backbone extracted and saved to '{output_path}'.")

    return backbone_df


import os
import pandas as pd


def save_meaningful_sample_with_sao_paulo(
        mobility_edges_path: str,
        output_path: str = "data/sample_mobility_edges.csv",
        n: int = 10
) -> pd.DataFrame:
    """
    Loads mobility edge data, filters meaningful flows, prioritizes São Paulo connections,
    samples n rows, and saves to CSV.

    Args:
        mobility_edges_path (str): Path to the Excel mobility data file.
        output_path (str): Path to save the sample CSV.
        n (int): Total number of rows in the final sample.

    Returns:
        pd.DataFrame: Sampled DataFrame with source/target codes, names, and flow.
    """

    # Load and select relevant columns
    df = pd.read_excel(mobility_edges_path)

    df = df[[
        'CODMUNDV_A', 'NOMEMUN_A', 'CODMUNDV_B', 'NOMEMUN_B', 'VAR05'
    ]].rename(columns={
        'CODMUNDV_A': 'source',
        'NOMEMUN_A': 'source_name',
        'CODMUNDV_B': 'target',
        'NOMEMUN_B': 'target_name',
        'VAR05': 'weekly_flow'
    })

    # Filter for meaningful flow
    df = df[df['weekly_flow'] > 0]

    # Prioritize São Paulo related edges
    sao_paulo_mask = df['source_name'].str.contains("são paulo", case=False, na=False) | \
                     df['target_name'].str.contains("são paulo", case=False, na=False)

    sao_paulo_rows = df[sao_paulo_mask].sample(n=min(3, len(df[sao_paulo_mask])), random_state=42)
    other_rows = df[~sao_paulo_mask].sample(n=max(0, n - len(sao_paulo_rows)), random_state=42)

    # Combine and save
    sample_df = pd.concat([sao_paulo_rows, other_rows]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_df.to_csv(output_path, index=False)

    print(f"[✓] Meaningful sample with São Paulo emphasis saved to '{output_path}'.")

    return sample_df

