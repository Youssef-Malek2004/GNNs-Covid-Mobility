import os
import pandas as pd


def load_and_save_covid_data(
    input_dir="data",
    output_path="data/covid_brazil_combined.csv"
) -> pd.DataFrame:
    """
    Load and concatenate COVID-19 data from multiple yearly CSV files.
    Saves the final DataFrame to disk to avoid reprocessing next time.

    Args:
        input_dir (str): Directory containing yearly COVID CSV files.
        output_path (str): Path to save or load the combined dataset.

    Returns:
        pd.DataFrame: The full sorted and cleaned COVID dataset.
    """
    if os.path.exists(output_path):
        print(f"[✓] Found saved COVID dataset at {output_path}. Loading it...")
        return pd.read_csv(output_path, parse_dates=["date"])

    print("[⚙] Processing raw yearly COVID files...")

    files = [
        "cases-brazil-cities-time_2020.csv",
        "cases-brazil-cities-time_2021.csv",
        "cases-brazil-cities-time_2022.csv",
        "cases-brazil-cities-time.csv",  # Jan–Mar 2023
    ]

    dfs = []
    for f in files:
        full_path = os.path.join(input_dir, f)
        if os.path.exists(full_path):
            dfs.append(pd.read_csv(full_path))
        else:
            print(f"[!] Warning: {f} not found in {input_dir}")

    if not dfs:
        raise FileNotFoundError("No COVID files found to load.")

    covid_df = pd.concat(dfs, ignore_index=True)
    covid_df['date'] = pd.to_datetime(covid_df['date'])
    covid_df = covid_df.sort_values(by=['ibgeID', 'date'])

    # Save for next time
    covid_df.to_csv(output_path, index=False)
    print(f"[✓] Saved combined COVID dataset to {output_path}")

    return covid_df
