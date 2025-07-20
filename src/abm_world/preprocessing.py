import csv
import pandas as pd
from pathlib import Path

RANKING_METHODS = ["morris", "random_forest"]

def process_parameters_from_csv(file_path: Path) -> pd.DataFrame:
    """
    Reads a CSV file containing parameter names and values, and returns a DataFrame.
    """
    df = pd.read_csv(file_path)
    
    df.columns = [c.strip() for c in df.columns]
    
    cols = [
        "Parameter Number",
        "Parameter Name",
        "Morris",
        "Random Forest",
        "Lower bound",
        "Upper bound",
        "Default Value"
    ]

    # Some columns may have extra spaces or slightly different names, so use fuzzy matching if needed
    df = df[[col for col in df.columns if any(key in col for key in cols)]]
    df = df.rename(columns=lambda x: x.strip())
    
    # Rename columns for consistency
    df = df.rename(columns={
        "Parameter Number": "parameter_number",
        "Parameter Name": "parameter_name",
        "Morris": "morris_ranking",
        "Random Forest": "random_forest_ranking",
        "Lower bound": "lower_bound",
        "Upper bound": "upper_bound",
        "Default Value": "default_value"
    })
    
    # Fill missing rankings with 9999 and convert to numeric
    for col in ["morris_ranking", "random_forest_ranking"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(9999).astype(int)
    
    # Return only the requested columns
    return df[["parameter_number", "parameter_name", "morris_ranking", "random_forest_ranking", "lower_bound", "upper_bound", "default_value"]]

def extract_n_parameters(df: pd.DataFrame, ranking_method: str, n: int) -> list[str]:
    """
    Extracts a list of the top n parameters' names based on the specified ranking method.
    Args:
        df: DataFrame containing parameter information.
        ranking_method: The ranking method to use (e.g., "morris", "random_forest").
        n: Number of top parameters to extract.
    """
    if ranking_method not in RANKING_METHODS:
        raise ValueError(f"Invalid ranking method: {ranking_method}. Choose from {RANKING_METHODS}.")
    ranking_col = f"{ranking_method}_ranking"
    df = df.sort_values(by=ranking_col).reset_index(drop=True)
    # Select only the top n parameters
    df = df.head(n)
    return df["parameter_name"].tolist()

def small_scaffold_adjustment(value: float) -> float:
    """
    Adjusts the value for the small scaffold experimental data.
    The adjustment is based on the formula:
    ((0.6)^3 / 1000 / 0.3 ) * value
    """
    return ((0.6 ** 3) / 1000 / 0.3) * value

def extract_small_scaffold_experimental(file_path: Path) -> pd.DataFrame:
    """
    Extracts experimental data from CSV file corresponding to the small scaffold.
    We apply the following formulas:
        - cell_number = ((0.6)^3 / 1000) / (0.3 * average of picogreen_cells)
        - collagen_pg = ((0.6)^3 / 1000) / (0.3 * average of sircol_collagen_ug) * 10^6
    """
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    # Calculate average for each config (GH2, GH5, GH10)
    df["picogreen_cells"] = df["picogreen_cells"].astype(float)
    df["sircol_collagen_ug"] = df["sircol_collagen_ug"].astype(float)
    df["bradford_protein_ug_per_ml"] = df["bradford_protein_ug_per_ml"].astype(float)

    # Add a column for live cells
    df["live_cells"] = df["picogreen_cells"] * (df["cell_viability"])

    # Calculate the average values for each group and time point
    averages = df.groupby(["group", "time_hour"]).mean().reset_index()

    # Apply scaffold adjustment to the mean live_cells
    averages["small_scaffold_cell_avg"] = averages["live_cells"].apply(small_scaffold_adjustment)

    # Collagen adjustment (convert from ug to pg)
    averages["small_scaffold_collagen_pg"] = averages["sircol_collagen_ug"].apply(
        lambda x: small_scaffold_adjustment(x) * 10e6
    )
    
    return averages

if __name__ == "__main__":
    print("===== PARAMETER PROCESSING =====")
    file_path = Path("input/sensitivity_analysis.csv")
    df = process_parameters_from_csv(file_path)
    print(df)
    print("\n===== TOP PARAMETERS =====")
    top_params = extract_n_parameters(df, "random_forest", 5)
    print(top_params)

    # Example usage
    print("===== EXPERIMENTAL DATA PROCESSING =====")
    file_path = Path("input/experimental.csv")
    df = extract_small_scaffold_experimental(file_path)
    print(df)







    