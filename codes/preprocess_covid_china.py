import pandas as pd
import os
import re

def load_and_print_city_name_mapping(filepath):
    """
    Loads a CSV mapping of Chinese city names to English names.
    Handles disambiguation for duplicate English names by appending province-level codes.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        Tuple[dict, dict]: Two dictionaries:
            - Chinese-to-English name mapping (disambiguated)
            - English-to-Chinese name mapping (inverse)
    """

    print(f"[✓] Loading city name mappings from: {filepath}")
    df = pd.read_csv(filepath)

    ch_names = df["City_CH"].tolist()
    en_names = df["City_EN"].tolist()

    ch_to_en = dict(zip(ch_names, en_names))
    en_to_ch = {}

    # Track duplicates
    duplicates = {}
    for ch, en in zip(ch_names, en_names):
        if en in en_to_ch:
            if en not in duplicates:
                duplicates[en] = [en_to_ch[en]]
            duplicates[en].append(ch)
        else:
            en_to_ch[en] = ch

    # Disambiguation dictionary (hardcoded based on known conflicts)
    disambiguation_map = {
        "Fuzhou": {"福州市": "Fuzhou_FJ", "抚州市": "Fuzhou_JX"},
        "Suzhou": {"苏州市": "Suzhou_JS", "宿州市": "Suzhou_AH"},
        "Taizhou": {"泰州市": "Taizhou_JS", "台州市": "Taizhou_ZJ"},
        "Yichun": {"伊春市": "Yichun_HLJ", "宜春市": "Yichun_JX"},
        "Yulin": {"玉林市": "Yulin_GX", "榆林市": "Yulin_SAX"},
    }

    for en, conflict_map in disambiguation_map.items():
        for ch, disamb_en in conflict_map.items():
            ch_to_en[ch] = disamb_en
            en_to_ch[disamb_en] = ch
        if en in en_to_ch:
            del en_to_ch[en]  # Remove ambiguous entry

    print(f"[✓] Total mappings: {len(ch_to_en)}")
    print(f"[✓] No missing Chinese names" if None not in ch_to_en else "[!] Missing values in CH names")
    print(f"[✓] No missing English names" if None not in ch_to_en.values() else "[!] Missing values in EN names")
    print(f"[✓] No duplicated Chinese names" if len(set(ch_to_en.keys())) == len(ch_to_en) else "[!] Duplicate CH names found")
    print(f"[✓] Duplicated English names: {len(duplicates)}")
    print("[✓] Sample mappings (CH -> EN):")
    for i, (ch, en) in enumerate(list(ch_to_en.items())[:10]):
        print(f"    {ch} -> {en}")

    return ch_to_en, en_to_ch

def load_filtered_baidu_mobility_files(data_dir, kind="in", city_ch_to_en=None, valid_chinese_cities=None):
    """
    Loads Baidu mobility matrices, maps city names to English, and filters matrices to only include valid cities.

    Args:
        data_dir (str): Path to the directory containing Baidu CSV files.
        kind (str): Either "in" or "out", indicating inflow or outflow data.
        city_ch_to_en (dict): Dictionary mapping Chinese city names to English.
        valid_chinese_cities (set): Set of valid Chinese city names to keep in the matrices.

    Returns:
        dict: A dictionary where keys are dates (YYYY-MM-DD) and values are filtered DataFrames.
    """

    kind = kind.lower()
    prefix = f"baidu_{kind}_"
    data = {}

    print(f"[✓] Scanning for Baidu {kind} files in: {data_dir}")

    for file in sorted(os.listdir(data_dir)):
        if file.startswith(prefix) and file.endswith(".csv"):
            match = re.search(rf"{prefix}(\d{{8}})", file)
            if not match:
                continue
            try:
                date_str = match.group(1)
                date = pd.to_datetime(date_str).strftime("%Y-%m-%d")

                df = pd.read_csv(os.path.join(data_dir, file), index_col=0)

                # Filter both rows and columns to valid Chinese cities only
                if valid_chinese_cities is not None:
                    df = df[df.index.isin(valid_chinese_cities)]
                    df = df.loc[:, df.columns.isin(valid_chinese_cities)]

                # Translate city names
                if city_ch_to_en:
                    df.columns = [city_ch_to_en.get(col, col) for col in df.columns]
                    df.index = [city_ch_to_en.get(idx, idx) for idx in df.index]

                data[date] = df

            except Exception as e:
                print(f"[!] Failed to load {file}: {e}")

    print(f"[✓] Loaded {len(data)} mobility matrices for kind='{kind}'")
    if data:
        sample_date = list(data.keys())[0]
        print(f"[✓] Sample matrix on {sample_date}:")
        print(data[sample_date].head())

    return data


import networkx as nx
import pandas as pd


def build_graph_from_mobility_matrix(matrix: pd.DataFrame) -> nx.DiGraph:
    """
    Builds a directed graph from a mobility matrix DataFrame.

    Args:
        matrix (pd.DataFrame): A square DataFrame where rows and columns are city names
                               and values are mobility flows (NaNs mean no edge).

    Returns:
        nx.DiGraph: A directed graph where each edge has a 'weight' attribute.
    """
    G = nx.DiGraph()
    cities = list(matrix.index)
    G.add_nodes_from(cities)

    for src in matrix.index:
        for dst in matrix.columns:
            weight = matrix.loc[src, dst]
            if pd.notna(weight):
                G.add_edge(src, dst, weight=weight)

    return G


def describe_graph(G: nx.DiGraph, max_samples: int = 10):
    """
    Prints a summary of the graph including number of nodes, edges, and sample entries.

    Args:
        G (nx.DiGraph): The graph to describe.
        max_samples (int): Number of sample nodes and edges to print.
    """
    print(f"[✓] Number of nodes: {len(G.nodes)}")
    print(f"[✓] Number of edges: {len(G.edges)}")

    print("\n[✓] Sample nodes:")
    print(list(G.nodes)[:max_samples])

    print("\n[✓] Sample edges (src -> dst : weight):")
    for u, v, d in list(G.edges(data=True))[:max_samples]:
        print(f"{u} -> {v} : {d['weight']}")

import numpy as np
import networkx as nx

import numpy as np

from collections import defaultdict
import numpy as np

def compute_average_mobility_matrix_with_threshold(mobility_data, threshold=0.9):
    """
    Computes an average mobility matrix using cities present in at least a threshold % of matrices.

    Args:
        mobility_data (dict): date -> DataFrame
        threshold (float): minimum fraction of matrices a city must appear in

    Returns:
        tuple: (avg_matrix, cities)
    """
    print(f"[✓] Computing average mobility matrix with threshold={threshold*100:.0f}%")

    city_counter = defaultdict(int)

    for df in mobility_data.values():
        for city in df.index.intersection(df.columns):
            city_counter[city] += 1

    min_count = int(len(mobility_data) * threshold)
    common_cities = [city for city, count in city_counter.items() if count >= min_count]
    common_cities = sorted(common_cities)

    if not common_cities:
        print("[!] No cities meet the threshold requirement.")
        return None, []

    stacked = np.zeros((len(common_cities), len(common_cities)))
    count = 0

    for df in mobility_data.values():
        try:
            df_filtered = df.loc[common_cities, common_cities].fillna(0)
            stacked += df_filtered.to_numpy()
            count += 1
        except:
            continue

    avg_matrix = stacked / count
    print(f"[✓] Averaged over {count} matrices")
    print(f"[✓] Number of common cities: {len(common_cities)}")
    return avg_matrix, common_cities




import networkx as nx

import networkx as nx
import numpy as np

def extract_nodes_and_edges_from_matrix(avg_matrix, cities, apply_backbone=True, alpha=0.01, top_k=5):
    """
    Constructs a directed graph from an average matrix, optionally extracting backbone using pij filtering and top-k neighbors.
    Ensures each city retains at least `top_k` strongest outgoing edges regardless of pij value.

    Returns:
        networkx.DiGraph: Graph with filtered weighted edges
    """
    import pandas as pd
    from collections import defaultdict
    import numpy as np
    import networkx as nx

    G = nx.DiGraph()

    # Convert to DataFrame if needed
    if isinstance(avg_matrix, np.ndarray):
        avg_matrix = pd.DataFrame(avg_matrix, index=cities, columns=cities)

    degree = defaultdict(int)
    strength = defaultdict(float)
    edges = []

    print(f"[•] Scanning matrix to gather all edges...")
    for i, src in enumerate(cities):
        for j, dst in enumerate(cities):
            if i != j:
                w = avg_matrix.iloc[i, j]
                if not np.isnan(w) and w > 0:
                    edges.append((i, j, w))
                    degree[i] += 1
                    strength[i] += w

    print(f"[✓] Total raw edges before filtering: {len(edges)}")
    zero_out_nodes = [cities[i] for i in range(len(cities)) if degree[i] == 0]
    print(f"[!] Cities with 0 outgoing edges in raw matrix: {len(zero_out_nodes)} → {zero_out_nodes[:5]}")

    if apply_backbone:
        print("[•] Applying backbone filtering (pij + top-k)...")
        edge_records = []
        for i, j, w in edges:
            si, ki = strength[i], degree[i]
            sj, kj = strength[j], degree[j]

            pij_i = (1 - w / si) ** (ki - 1) if ki > 1 and si > 0 else 1
            pij_j = (1 - w / sj) ** (kj - 1) if kj > 1 and sj > 0 else 1
            pij = min(pij_i, pij_j)

            edge_records.append({
                "src": cities[i],
                "dst": cities[j],
                "weight": w,
                "pij": pij
            })

        edge_df = pd.DataFrame(edge_records)

        # Always keep top-k strongest outgoing edges
        topk_keep = set()
        for node in cities:
            node_edges = edge_df[edge_df["src"] == node]
            topk = node_edges.nlargest(top_k, "weight")
            topk_keep.update(zip(topk["src"], topk["dst"]))

        edge_df["keep_topk"] = edge_df.apply(lambda row: (row["src"], row["dst"]) in topk_keep, axis=1)
        edge_df["keep"] = (edge_df["pij"] < alpha) | edge_df["keep_topk"]

        kept_edges = edge_df[edge_df["keep"]]
        print(f"[✓] Edges after filtering: {len(kept_edges)}")
        print(f"[✓] pij < alpha: {(edge_df['pij'] < alpha).sum()}, top-{top_k} edges kept: {edge_df['keep_topk'].sum()}")

        unique_edges = set()
        for _, row in kept_edges.iterrows():
            edge = (row["src"], row["dst"])
            if edge not in unique_edges:
                G.add_edge(*edge, weight=row["weight"])
                unique_edges.add(edge)

        print(f"✓ Actually added {len(unique_edges)} unique edges.")



    else:
        for i, j, w in edges:
            G.add_edge(cities[i], cities[j], weight=w)

    # Ensure all cities exist as nodes
    G.add_nodes_from(cities)

    # Check for isolated nodes
    isolated = list(nx.isolates(G))
    print(f"[!] Isolated cities (no mobility edges): {len(isolated)} → {isolated[:5]}")

    print(f"[✓] Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G



def inspect_mobility_data(mobility_data):
    """
    Inspect the loaded mobility matrices: report shapes and provide examples.
    Also, show an example of one empty matrix.
    """
    print(f"[✓] Inspected {len(mobility_data)} matrices.")

    shapes = [(df.shape if isinstance(df, pd.DataFrame) else None) for df in mobility_data.values()]
    unique_shapes = list(set(shapes))
    print(f"[✓] Unique matrix shapes: {unique_shapes}")

    # Display a few examples
    for date in list(mobility_data.keys())[:5]:
        print(f"    {date}: {mobility_data[date].shape}")

    # Count occurrences of each shape
    from collections import Counter
    shape_counts = Counter(shapes)
    print("\n[✓] Matrix shape counts:")
    for shape, count in shape_counts.items():
        print(f"    {shape}: {count} matrices")

    # Show one empty file
    for date, df in mobility_data.items():
        if df.shape == (0, 0):
            print(f"\n[!] Example of empty matrix: {date}")
            break


import pandas as pd
import re

import pandas as pd
from difflib import get_close_matches

def fuzzy_match_city(name, valid_cities, cutoff=0.8):
    """
    Tries to find the best match for a city name using prefix or fuzzy match.
    """
    matches = get_close_matches(name, valid_cities, n=1, cutoff=cutoff)
    return matches[0] if matches else None

from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_and_filter_covid_timeseries(file_path, valid_english_cities, verbose=True):
    """
    Loads and filters COVID-19 case time series using fuzzy matching on city names,
    and applies Z-score normalization (per city).

    Args:
        file_path (str): Path to the Excel file.
        valid_english_cities (set): Set of English city names from the mobility graph.
        verbose (bool): Whether to print matching summary.

    Returns:
        pd.DataFrame: Normalized time series with matched and cleaned city names as index.
    """
    print(f"[✓] Loading COVID-19 case data from: {file_path}")
    df = pd.read_excel(file_path)

    # Drop the city code column (2nd column)
    df = df.drop(columns=df.columns[1])

    # Extract city names
    raw_cities = df.iloc[:, 0].astype(str)

    # Match to valid city names
    matched_names = {}
    for name in raw_cities:
        match = fuzzy_match_city(name, valid_english_cities)
        if match:
            matched_names[name] = match

    # Filter and rename
    df = df[df[df.columns[0]].isin(matched_names.keys())].copy()
    df[df.columns[0]] = df[df.columns[0]].map(matched_names)

    # Set city name as index
    df = df.set_index(df.columns[0])

    # Keep only time series columns (from 3rd column onward)
    df = df.iloc[:, 1:]

    # Convert to numeric and handle missing values
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Apply Z-score normalization (per city)
    scaler = StandardScaler()
    df_z = pd.DataFrame(
        scaler.fit_transform(df.T).T,  # transpose to scale across time (per city)
        index=df.index,
        columns=df.columns
    ).astype("float32")
    df_z.index.name = "City_name"

    if verbose:
        unmatched = set(raw_cities) - set(matched_names.keys())
        print(f"[✓] Matched cities: {len(matched_names)}")
        print(f"[!] Unmatched cities: {len(unmatched)}")
        if unmatched:
            print(f"[!] Example unmatched: {list(unmatched)[:10]}")
        print(f"[✓] Final shape: {df_z.shape}")
        print(f"[✓] Date range: {df_z.columns[0]} to {df_z.columns[-1]}")
        print(f"[✓] Example matched cities: {df_z.index[:5].tolist()}")

    return df_z


import torch
import numpy as np
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal


def build_temporal_graph_data(edge_index, features_df, window_size=10):
    """
    Builds a DynamicGraphTemporalSignal object from a time series DataFrame and edge index.

    Args:
        edge_index (np.array): Shape [2, num_edges], edge list of the static graph.
        features_df (pd.DataFrame): Shape [num_nodes, num_timesteps], z-score normalized time series.
        window_size (int): Length of input sequence for GCRN (e.g. 10 days).

    Returns:
        DynamicGraphTemporalSignal: Ready to use in GCRN.
    """
    features_np = features_df.to_numpy()  # shape: [num_nodes, num_timesteps]
    num_nodes, num_timesteps = features_np.shape

    edge_index_np = np.array(edge_index)  # already in [2, num_edges] format

    snapshots = []
    for t in range(num_timesteps - window_size):
        # For each timestep, features are current snapshot
        X = features_np[:, t]  # shape: [num_nodes]
        Y = features_np[:, t + 1]  # Predict next step

        X = X.reshape(-1, 1)  # shape: [num_nodes, 1]
        Y = Y.reshape(-1, 1)

        snapshots.append((X, Y))

    # Unpack into lists
    features_list = [x for x, _ in snapshots]
    targets_list = [y for _, y in snapshots]

    dataset = DynamicGraphTemporalSignal(
        edge_index_list=[edge_index_np] * len(features_list),
        edge_weight_list=None,
        features=features_list,
        targets=targets_list,
    )

    return dataset






