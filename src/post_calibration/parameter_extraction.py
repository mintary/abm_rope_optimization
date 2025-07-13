import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

def extract_parameters_from_csv(
    csv_file_path: Union[str, Path], 
    num_parameters: int,
    top_n: int = 1
) -> Dict[str, Any]:
    """
    Extract the best parameter set(s) from a ROPE optimization CSV file.
    
    In the SPOTPY ROPE algorithm, we maximize the objective function value
    (which is the negative of our error function). So the best parameter set
    has the highest value in the first column.
    
    Args:
        csv_file_path: Path to the ROPE CSV file
        top_n: Number of top parameter sets to return (default: 1)
        
    Returns:
        Dictionary with best parameter values and metadata
    """
    if isinstance(csv_file_path, str):
        csv_file_path = Path(csv_file_path)
    
    if not csv_file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    df = pd.read_csv(csv_file_path, header=None)
    
    # The first column contains the objective function value (negative error)
    # Higher values are better (less error)
    
    # Sort by objective function value (descending)
    # Use integer index as column name when sorting
    df_sorted = df.sort_values(by=df.columns[0], ascending=False)
    
    # Get the top N rows
    top_params = df_sorted.head(top_n)
    
    # Extract parameter values
    result = {
        "best_parameter_sets": [],
        "objective_function_values": [],
        "actual_error_values": []
    }
    
    for i, row in top_params.iterrows():
        obj_value = float(row[0])
        
        # Actual error is negative of objective function value
        actual_error = -obj_value
        
        # Parameter values start from the second column
        param_values = [float(v) for v in row[1:num_parameters + 1]]
        
        # Add to results
        result["best_parameter_sets"].append(param_values)
        result["objective_function_values"].append(obj_value)
        result["actual_error_values"].append(actual_error)
    
    # If only one parameter set was requested, simplify the output
    if top_n == 1:
        result["best_parameter_set"] = result["best_parameter_sets"][0]
        result["objective_function_value"] = result["objective_function_values"][0]
        result["actual_error_value"] = result["actual_error_values"][0]
    
    return result


def save_best_parameters_to_json(
    csv_file_path: Union[str, Path],
    num_parameters: int,
    output_path: Optional[Union[str, Path]] = None,
    top_n: int = 10
) -> Path:
    """
    Extract best parameters from CSV and save to a JSON file.
    
    Args:
        csv_file_path: Path to the ROPE CSV file
        output_path: Path to save the JSON output (default: output/best_parameters.json)
        top_n: Number of top parameter sets to include
        
    Returns:
        Path to the saved JSON file
    """
    if output_path is None:
        output_path = Path("output/best_parameters.json")
    elif isinstance(output_path, str):
        output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract parameters
    best_params = extract_parameters_from_csv(csv_file_path, num_parameters, top_n=top_n)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    return output_path


if __name__ == "__main__":
    # Example usage
    csv_path = Path("output/rope_abm_optimization.csv")

    num_params = 5
    
    # Extract and display best parameter set
    best = extract_parameters_from_csv(csv_path, num_params, 1)
    print(f"Best parameter set: {best['best_parameter_set']}")
    print(f"Objective function value: {best['objective_function_value']}")
    print(f"Actual error value: {best['actual_error_value']}")
    
    # Save top 10 parameter sets to JSON
    json_path = save_best_parameters_to_json(csv_path, num_params, top_n=10)
    print(f"Saved top 10 parameter sets to {json_path}")