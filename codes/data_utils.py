import numpy as np
import torch
import pandas as pd

def prepare_temporal_graph_data_non_overlapping(
    filtered_covid_df: pd.DataFrame,
    sequence_length: int = 15,
    feature_column: str = "z_newCases",
    device: torch.device = torch.device("cpu"),
    train_split: float = 0.8
):
    """
    Converts filtered COVID data into GCRN-ready non-overlapping temporal sequences (X, Y).

    Args:
        filtered_covid_df (pd.DataFrame): DataFrame with columns ['date', 'ibgeID', feature_column]
        sequence_length (int): Length of the input window (T).
        feature_column (str): Feature to extract per city per day.
        device (torch.device): Device to put the tensors on.
        train_split (float): Percentage of data to allocate for training.

    Returns:
        X_train, X_test, Y_train, Y_test (torch.Tensor): Ready for GCRN input.
    """

    # Pivot: rows = dates, columns = cities (shape: [num_days, num_nodes])
    pivot_df = filtered_covid_df.pivot(index='date', columns='ibgeID', values=feature_column).sort_index()
    data_array = pivot_df.fillna(0).to_numpy()
    num_days, num_nodes = data_array.shape

    X_list, Y_list = [], []

    i = 0
    while i + sequence_length < num_days - 1:
        x_seq = data_array[i:i + sequence_length]        # [T, N]
        y_target = data_array[i + sequence_length]       # [N]

        x_seq = np.expand_dims(x_seq, axis=-1)           # [T, N, 1]
        y_target = np.expand_dims(y_target, axis=-1)     # [N, 1]

        X_list.append(x_seq)
        Y_list.append(y_target)

        i += sequence_length + 1  # Jump forward without overlapping

    # Stack and convert to tensors
    X = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32).to(device)

    # Train/test split
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"[📉] (Non-overlapping) X shape: {X.shape} | Y shape: {Y.shape}")
    print(f"[📉] Train: {X_train.shape} | Test: {X_test.shape}")

    return X_train, X_test, Y_train, Y_test


def prepare_temporal_graph_data(
    filtered_covid_df: pd.DataFrame,
    sequence_length: int = 15,
    feature_column: str = "z_newCases",
    device: torch.device = torch.device("cpu"),
    train_split: float = 0.8
):
    """
    Converts filtered COVID data into GCRN-ready temporal sequences (X, Y).

    Args:
        filtered_covid_df (pd.DataFrame): DataFrame with columns ['date', 'ibgeID', feature_column]
        sequence_length (int): Length of the input window (T).
        feature_column (str): Feature to extract per city per day.
        device (torch.device): Device to put the tensors on.
        train_split (float): Percentage of data to allocate for training.

    Returns:
        X_train, X_test, Y_train, Y_test (torch.Tensor): Ready for GCRN input.
    """

    # Pivot: rows = dates, columns = cities (shape: [num_days, num_nodes])
    pivot_df = filtered_covid_df.pivot(index='date', columns='ibgeID', values=feature_column).sort_index()

    # Fill missing with 0 and convert to numpy
    data_array = pivot_df.fillna(0).to_numpy()
    num_days, num_nodes = data_array.shape

    X_list, Y_list = [], []

    for i in range(num_days - sequence_length):
        x_seq = data_array[i:i + sequence_length]      # [T, N]
        y_target = data_array[i + sequence_length]     # [N]

        x_seq = np.expand_dims(x_seq, axis=-1)         # [T, N, 1]
        y_target = np.expand_dims(y_target, axis=-1)   # [N, 1]

        X_list.append(x_seq)
        Y_list.append(y_target)

    # Stack and convert to tensors
    X = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32).to(device)

    # Train/test split
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"[✓] X shape: {X.shape} | Y shape: {Y.shape}")
    print(f"[✓] Train: {X_train.shape} | Test: {X_test.shape}")

    return X_train, X_test, Y_train, Y_test


def generate_sliding_temporal_graph_data(
    filtered_covid_df: pd.DataFrame,
    input_window: int = 15,
    output_window: int = 7,
    feature_column: str = "z_newCases",
    device: torch.device = torch.device("cpu"),
    train_split: float = 0.8
):
    """
    Generates X and Y sequences using a sliding window over the COVID time series.

    Args:
        filtered_covid_df (pd.DataFrame): DataFrame with ['date', 'ibgeID', feature_column].
        input_window (int): Number of days to use as input.
        output_window (int): Number of days to predict.
        feature_column (str): The column used as feature.
        device (torch.device): CPU/GPU.
        train_split (float): Ratio of training data.

    Returns:
        Tuple of torch.Tensors: (X_train, X_test, Y_train, Y_test)
            - X: [num_samples, input_window, num_nodes, 1]
            - Y: [num_samples, output_window, num_nodes, 1]
    """
    # Pivot the data to [num_days, num_nodes]
    pivot_df = filtered_covid_df.pivot(index='date', columns='ibgeID', values=feature_column).sort_index()
    data_array = pivot_df.fillna(0).to_numpy()

    num_days, num_nodes = data_array.shape
    X_list, Y_list = [], []

    max_index = num_days - input_window - output_window + 1

    for i in range(max_index):
        x_seq = data_array[i:i + input_window]              # [input_window, num_nodes]
        y_seq = data_array[i + input_window:i + input_window + output_window]  # [output_window, num_nodes]

        # Add channel dimension
        x_seq = np.expand_dims(x_seq, axis=-1)              # [input_window, num_nodes, 1]
        y_seq = np.expand_dims(y_seq, axis=-1)              # [output_window, num_nodes, 1]

        X_list.append(x_seq)
        Y_list.append(y_seq)

    # Convert to tensors
    X = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32).to(device)

    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"[✓] Sliding window: X {X.shape}, Y {Y.shape}")
    print(f"[✓] Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, Y_train, Y_test

