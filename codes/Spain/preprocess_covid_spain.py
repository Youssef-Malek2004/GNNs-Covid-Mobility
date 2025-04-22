import os
import pandas as pd
from typing import Optional, Set

def zscore_safe(x):
    std = x.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([0] * len(x), index=x.index)
    return (x - x.mean()) / std


def filter_and_scale_spanish_covid_by_centrality(
    covid_df: pd.DataFrame,
    centrality_path: str = "data/Spain/centrality_provinces.csv",
    output_path: str = "data/Spain/filtered_scaled_covid.csv",
    city_whitelist: Optional[Set[int]] = None
) -> pd.DataFrame:
    """
    Filters and normalizes Spanish COVID-19 data by province code using Z-score normalization.

    Args:
        covid_df (pd.DataFrame): Raw Spanish COVID dataset.
        centrality_path (str): Path to CSV with 'Codmundv' column.
        output_path (str): File path to save or load the processed data.
        city_whitelist (Optional[Set[int]]): Subset of province codes to include.

    Returns:
        pd.DataFrame: Filtered and Z-score normalized COVID data.
    """
    if os.path.exists(output_path):
        print(f"[✓] Found saved preprocessed COVID data at '{output_path}'. Loading...")
        return pd.read_csv(output_path, parse_dates=['Fecha'])

    print("[⚙] Processing Spanish COVID data based on centrality filtering...")

    # Load centrality data
    centrality_df = pd.read_csv(centrality_path)
    centrality_city_ids = set(centrality_df['Codmundv'].dropna().astype(int).unique())
    valid_city_ids = centrality_city_ids & city_whitelist if city_whitelist else centrality_city_ids

    # Filter COVID data
    filtered_covid_df = covid_df[covid_df['cod_ine'].isin(valid_city_ids)].copy()
    print(f"[✓] Filtered to {filtered_covid_df['cod_ine'].nunique()} provinces, {len(filtered_covid_df):,} rows.")

    # Clip negative values (if any)
    filtered_covid_df['Casos'] = filtered_covid_df['Casos'].clip(lower=0)

    # Z-score normalization of daily cases per province
    filtered_covid_df['z_newCases'] = filtered_covid_df.groupby('cod_ine')['Casos'].transform(zscore_safe)
    print("[✓] Applied Z-score normalization on daily cases (Casos).")

    # Save
    filtered_covid_df.to_csv(output_path, index=False)
    print(f"[✓] Saved filtered + scaled Spanish COVID data to '{output_path}'.")

    return filtered_covid_df
