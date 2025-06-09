import os
import pandas as pd
from typing import Optional, Set


def zscore_safe(x):
    std = x.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([0] * len(x), index=x.index)
    return (x - x.mean()) / std


def filter_and_scale_covid_by_centrality(
    covid_df: pd.DataFrame,
    centrality_path: str = "data/Centrality_indices.xlsx",
    population_path: str = "data/cleaned_population_2022.csv",
    output_path: str = "data/filtered_scaled_covid.csv",
    city_whitelist: Optional[Set[int]] = None
) -> pd.DataFrame:
    """
    Filters COVID-19 DataFrame to only include municipalities in the centrality dataset,
    and applies Z-score normalization on newCases and newDeaths per city.
    Saves and loads the result to avoid reprocessing.

    Args:
        covid_df (pd.DataFrame): Full COVID dataset.
        centrality_path (str): Path to the centrality Excel file with 'Codmundv'.
        output_path (str): File path to save or load the preprocessed dataset.
        city_whitelist (Optional[Set[int]]):
    Returns:
        pd.DataFrame: Filtered and scaled COVID DataFrame.
    """
    # Load centrality data
    centrality_df = pd.read_excel(centrality_path)
    centrality_city_ids = set(centrality_df['Codmundv'].dropna().astype(int).unique())
    valid_city_ids = centrality_city_ids & city_whitelist if city_whitelist else centrality_city_ids

    # Filter COVID data
    filtered_covid_df = covid_df[covid_df['ibgeID'].isin(valid_city_ids)].copy()
    print(f"[✓] Filtered to {filtered_covid_df['ibgeID'].nunique()} cities, {len(filtered_covid_df):,} rows.")

    # Clip negative values to zero
    filtered_covid_df['newCases'] = filtered_covid_df['newCases'].clip(lower=0)
    filtered_covid_df['newDeaths'] = filtered_covid_df['newDeaths'].clip(lower=0)
    print("[✓] Negative values in 'newCases' and 'newDeaths' clipped to 0.")

    # Load and merge population data
    pop_df = pd.read_csv(population_path)
    filtered_covid_df = filtered_covid_df.merge(pop_df[['ibgeID', 'population']], on='ibgeID', how='left')

    # Normalize per 100k inhabitants
    filtered_covid_df['cases_per_100k'] = (filtered_covid_df['newCases'] / filtered_covid_df['population']) * 1e5
    filtered_covid_df['deaths_per_100k'] = (filtered_covid_df['newDeaths'] / filtered_covid_df['population']) * 1e5
    print("[✓] Computed cases and deaths per 100,000 population.")

    # Z-score normalization
    filtered_covid_df['z_newCases'] = filtered_covid_df.groupby('ibgeID')['newCases'].transform(zscore_safe)
    filtered_covid_df['z_newDeaths'] = filtered_covid_df.groupby('ibgeID')['newDeaths'].transform(zscore_safe)
    print("[✓] Applied Z-score normalization.")

    # Save for future runs
    filtered_covid_df.to_csv(output_path, index=False)
    print(f"[✓] Saved filtered + scaled COVID data to '{output_path}'.")

    return filtered_covid_df

def invert_zscore_normalization(
    scaled_df: pd.DataFrame,
    original_df: pd.DataFrame,
    value_col: str = "z_newCases",
    original_value_col: str = "newCases",
    id_col: str = "ibgeID"
) -> pd.DataFrame:
    """
    Inverts z-score normalization on a per-city basis using original_df stats.

    Args:
        scaled_df (pd.DataFrame): The DataFrame with z-scored values.
        original_df (pd.DataFrame): The original DataFrame with real values.
        value_col (str): Column in scaled_df to invert (e.g., 'z_newCases').
        original_value_col (str): Column in original_df used to calculate mean and std (e.g., 'newCases').
        id_col (str): Column name that identifies the city (e.g., 'ibgeID').

    Returns:
        pd.DataFrame: A copy of scaled_df with an added column for unscaled values.
    """
    # Compute mean and std per city from original_df
    stats = original_df.groupby(id_col)[original_value_col].agg(['mean', 'std']).reset_index()
    stats.columns = [id_col, 'mean', 'std']

    # Merge stats back into the scaled_df
    merged_df = scaled_df.merge(stats, on=id_col, how='left')

    # Invert the z-score normalization
    unscaled_col = value_col.replace("z_", "")  # e.g., 'z_newCases' → 'newCases'
    merged_df[unscaled_col + "_reconstructed"] = merged_df[value_col] * merged_df['std'] + merged_df['mean']

    # Optional: drop the mean and std columns if not needed
    merged_df.drop(['mean', 'std'], axis=1, inplace=True)

    return merged_df


def clean_ibge_population_data(
        input_path: str = "data/CD2022_Collected_Imputed_Population_and_Total_Municipality_and_State_20231222.xlsx",
        output_path: str = "data/cleaned_population_2022.csv"
) -> pd.DataFrame:
    # Load Excel file
    xls = pd.ExcelFile(input_path)

    # Parse "Municípios" sheet
    df_raw = xls.parse("Municípios")

    # Drop metadata rows
    df = df_raw.iloc[2:].copy()
    df.columns = [
        "index", "UF", "COD_UF", "COD_MUNIC", "NOME_MUNICIPIO",
        "POP_COLETADA", "POP_IMPUTADA", "POP_TOTAL"
    ]

    # Ensure population is numeric
    df["POP_TOTAL"] = pd.to_numeric(df["POP_TOTAL"], errors="coerce").astype("Int64")

    # Build IBGE code: COD_UF + COD_MUNIC
    df["COD_UF"] = df["COD_UF"].astype(str).str.zfill(2)
    df["COD_MUNIC"] = df["COD_MUNIC"].astype(str).str.zfill(5)
    df["ibgeID"] = pd.to_numeric(df["COD_UF"] + df["COD_MUNIC"], errors="coerce").astype("Int64")

    # Keep only relevant rows
    df_clean = df[["ibgeID", "NOME_MUNICIPIO", "POP_TOTAL"]].rename(columns={
        "NOME_MUNICIPIO": "city_name",
        "POP_TOTAL": "population"
    }).dropna(subset=["ibgeID", "population"])

    # Save to CSV
    df_clean.to_csv(output_path, index=False)
    print(f"[✓] Cleaned population data saved to: {output_path}")

    return df_clean

