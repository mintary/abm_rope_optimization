import re
import json
import pandas as pd
from typing import Any

def extract_best_parameters(log_file_path):
    # Read the log file
    with open(log_file_path, 'r') as file:
        log_content = file.read()
    
    # Extract all iterations with their errors and parameters
    pattern = r"Iteration (\d+): Error (\d+\.\d+).*Parameters: \[([\d\., ]+)\]"
    iterations = re.findall(pattern, log_content)
    
    # Also capture NEW BEST entries
    best_pattern = r"NEW BEST at iteration (\d+): Error (\d+\.\d+)"
    best_iterations = re.findall(best_pattern, log_content)
    
    # Create a DataFrame with all iterations
    data = []
    for iter_num, error, params in iterations:
        param_list = [float(p.strip()) for p in params.split(',')]
        data.append({
            'Iteration': int(iter_num),
            'Error': float(error),
            'Parameters': param_list,
            'Param1': param_list[0] if len(param_list) > 0 else None,
            'Param2': param_list[1] if len(param_list) > 1 else None,
            'Param3': param_list[2] if len(param_list) > 2 else None,
            'Param4': param_list[3] if len(param_list) > 3 else None,
            'Param5': param_list[4] if len(param_list) > 4 else None,
        })
    
    df = pd.DataFrame(data)
    
    # Find the best result (minimum error)
    best_row = df.loc[df['Error'].idxmin()]
    
    return {
        'best_parameters': best_row['Parameters'],
        'best_error': best_row['Error'],
        'best_iteration': best_row['Iteration'],
        'all_iterations': df,
        'improvement': ((df['Error'].iloc[0] - best_row['Error']) / df['Error'].iloc[0]) * 100
    }


def save_best_parameters_to_json(results: dict[str, Any], output_path='output/best_parameters.json'):
    """ Save the best parameters and their details to a JSON file."""
    with open('output/best_parameters.json', 'w') as f:
        json.dump({
            'parameters': results['best_parameters'],
            'error': float(results['best_error']),
            'iteration': int(results['best_iteration']),
            'improvement': float(results['improvement'])
        }, f, indent=2)