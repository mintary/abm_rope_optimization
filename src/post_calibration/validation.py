"""
Validation module for ABM model with optimized parameters.

This module provides functionality to validate the ABM model by:
1. Running simulations with optimized parameters
2. Comparing results to validation data (day 9, time_hour=216)
3. Running multiple simulations per configuration to account for stochasticity
4. Calculating validation metrics
"""

import json
import uuid
import shutil
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Mapping
import csv
import subprocess
import platform
import os
from scipy import stats

from abm_world.simulation import prepare_sample, run_abm_simulation
from abm_world.preprocessing import extract_small_scaffold_experimental
import calibration.constants as constants

from calibration.error_function import normalized_biomarker_error

logger = logging.getLogger(__name__)

validation_logger = logging.getLogger(f"{__name__}.validation")
validation_logger.setLevel(logging.INFO)

if not validation_logger.handlers:
    log_dir = Path("output/validation")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / "validation.log", mode='w')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - VALIDATION - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    validation_logger.addHandler(file_handler)
    validation_logger.propagate = False  # Don't propagate to root logger


def run_multiple_validations(
        parameter_values: List[float],
        subprocess_run_dir: Path,
        bin_dir: Path,
        config_file_dir: Path,
        config_files: List[str],
        runs_per_config: int = 3,
        num_ticks: int = 289,
        tracked_biomarkers: List[str] = ["TotalFibroblast", "Collagen"],
        tracked_ticks: List[int] = [constants.TICKS_PER_DAY * 3, constants.TICKS_PER_DAY * 6, constants.TICKS_PER_DAY * 9],
) -> Dict[str, Dict[int, Dict[str, List[float]]]]:
    """
    Run multiple validation simulations for each configuration to account for stochasticity.
    
    Args:
        parameter_values: Optimized parameter values to use
        subprocess_run_dir: Directory for simulation runs
        bin_dir: Directory containing ABM binaries
        config_file_dir: Directory containing configuration files
        config_files: List of configuration files to use
        runs_per_config: Number of runs per configuration
        num_ticks: Total number of ticks for simulation
        tracked_biomarkers: List of biomarkers to track
        tracked_ticks: Ticks at which to track biomarkers
        
    Returns:
        Dictionary with results for each configuration, tick, and biomarker
        Structure: {config_name: {tick: {biomarker: [run1_value, run2_value, ...]}}}
    """
    validation_logger.info(f"Starting validation with parameters: {parameter_values}")
    validation_logger.info(f"Running {runs_per_config} simulations for each of {len(config_files)} configurations")
    
    results: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    
    # Create validation output directory
    validation_dir = Path("output/validation")
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    for config_file_name in config_files:
        config_name = config_file_name.replace("config_", "").replace(".txt", "")
        validation_logger.info(f"Starting validation for configuration: {config_name}")
        
        config_results: Dict[int, Dict[str, List[float]]] = {}
        
        for run_idx in range(runs_per_config):
            validation_logger.info(f"  Run {run_idx+1}/{runs_per_config} for {config_name}")
            
            run_id = f"validation_{config_name}"
            run_dir = subprocess_run_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            run_bin_dir = run_dir / "bin"
            run_bin_dir.mkdir(parents=True, exist_ok=True)
            for item in bin_dir.iterdir():
                if item.is_file():
                    dest = run_bin_dir / item.name
                    shutil.copy(item, dest)
            
            run_config_dir = run_dir / "configFiles"
            run_config_dir.mkdir(parents=True, exist_ok=True)
            source_config = config_file_dir / config_file_name
            dest_config = run_config_dir / config_file_name
            shutil.copy(source_config, dest_config)
            
            sample_path = run_dir / "Sample.txt"
            prepare_sample(parameter_values, sample_path)
            
            biomarker_output_dir = run_dir / "output"
            biomarker_output_dir.mkdir(parents=True, exist_ok=True)
            bin_stderr_path = biomarker_output_dir / "ABM_simulation_stderr.txt"
            bin_stdout_path = biomarker_output_dir / "ABM_simulation_stdout.txt"
            
            sim_results = run_abm_simulation(
                bin_path=run_bin_dir,
                bin_stderr_path=bin_stderr_path,
                bin_stdout_path=bin_stdout_path,
                sample_path=sample_path,
                config_path=dest_config,
                biomarker_output_dir=biomarker_output_dir,
                tracked_biomarkers=tracked_biomarkers,
                num_ticks=num_ticks,
                tracked_ticks=tracked_ticks,
                cwd=run_dir,
            )
            
            for tick, tick_data in sim_results.items():
                if tick not in config_results:
                    config_results[tick] = {biomarker: [] for biomarker in tracked_biomarkers}
                
                for biomarker, value in tick_data.items():
                    config_results[tick][biomarker].append(value)
        
        results[config_name] = config_results
    
    save_validation_results(results, validation_dir / "validation_results.json")
    
    return results


def save_validation_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save validation results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_logger.info(f"Saving validation results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    validation_logger.info(f"Validation results saved to {output_path}")

def calculate_validation_metrics(
        simulation_results: Dict[str, Dict[int, Dict[str, List[float]]]],
        experimental_data: Dict[str, Dict[int, Dict[str, float]]],
        simulation_validation_tick: int = constants.TICKS_PER_DAY * 9,
        experimental_validation_hour: int = 216,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate validation metrics comparing simulation results to experimental data.

    The form of the data for comparison looks something like this: 
    ```
    {
        "config_name": {
            144: {
                "Collagen": value (or list of values),
                "TotalFibroblast": value (or list of values),
            }
        }
    }
    ```
    Args:
        simulation_results: Simulation results from multiple runs
        experimental_data: Experimental data
        simulation_validation_tick: Tick to use for validation (default: day 9)
        experimental_validation_hour: Hour to use for validation (default: 216)
        
    Returns:
        Dictionary with validation metrics for each configuration and biomarker
    """
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    validation_logger.info(f"Calculating validation metrics for tick {simulation_validation_tick}")

    for config_name, config_results in simulation_results.items():
        if config_name not in metrics:
            metrics[config_name] = {}

        if simulation_validation_tick not in config_results:
            validation_logger.warning(f"Validation tick {simulation_validation_tick} not found in results for {config_name}")
            continue

        if config_name not in experimental_data or experimental_validation_hour not in experimental_data[config_name]:
            validation_logger.warning(f"No experimental data for {config_name} at hour {experimental_validation_hour}")
            continue

        sim_data = config_results[simulation_validation_tick]
        exp_data = experimental_data[config_name][experimental_validation_hour]

        for biomarker in sim_data:
            if biomarker not in metrics[config_name]:
                metrics[config_name][biomarker] = {}
            
            sim_values = sim_data[biomarker]
            exp_value = exp_data[biomarker]
            
            mean_value: float = float(np.mean(sim_values))
            std_value: float = float(np.std(sim_values))
            
            absolute_error = abs(mean_value - exp_value)
            relative_error = absolute_error / exp_value if exp_value != 0 else float('inf')
            
            confidence = 0.95
            n = len(sim_values)
            sem = std_value / np.sqrt(n)
            ci = stats.t.interval(confidence, n-1, loc=mean_value, scale=sem)
            
            # Store metrics
            metrics[config_name][biomarker] = {
                'mean': mean_value,
                'std': std_value,
                'experimental': exp_value,
                'absolute_error': absolute_error,
                'relative_error': relative_error,
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'within_ci': ci[0] <= exp_value <= ci[1]
            }
    
    return metrics

def convert_df_to_dict(df: pd.DataFrame):
    """
    Convert an experimental data DataFrame to a dictionary format suitable for validation.
    This has the form:
    {
        "config_name": {
            144: {
                "Collagen": value,
                "TotalFibroblast": value,
            }
        }
    }
    Args:
        df (pd.DataFrame): DataFrame with columns: group, time_hour, (raw values), small_scaffold_cell_avg, small_scaffold_collagen_avg
    """
    exp_data_dict: Dict[str, Dict[int, Dict[str, float]]] = {}
    
    for _, row in df.iterrows():
        group = row['group']
        time_hour = int(row['time_hour'])

        cell_count = float(round(row['small_scaffold_cell_avg']))
        collagen = float(row['small_scaffold_collagen_pg'])

        if group not in exp_data_dict:
            exp_data_dict[group] = {}
        
        if time_hour not in exp_data_dict[group]:
            exp_data_dict[group][time_hour] = {}
        
        exp_data_dict[group][time_hour]["TotalFibroblast"] = cell_count
        exp_data_dict[group][time_hour]["Collagen"] = collagen
    
    return exp_data_dict
        

def run_validation(
        parameter_values: List[float],
        subprocess_run_dir: Path,
        bin_dir: Path,
        config_file_dir: Path,
        experimental_data_file: Path,
        config_files: List[str] = ["config_Scaffold_GH2.txt", "config_Scaffold_GH5.txt", "config_Scaffold_GH10.txt"],
        runs_per_config: int = 3,
        output_dir: Path = Path("output/validation"),
) -> Dict[str, Any]:
    """
    Run complete validation workflow for the optimized parameters.
    
    Args:
        parameter_values: Optimized parameter values
        subprocess_run_dir: Directory for simulation runs
        bin_dir: Directory containing ABM binaries
        config_file_dir: Directory containing configuration files
        experimental_data_file: Path to experimental data
        config_files: List of configuration files to use
        runs_per_config: Number of runs per configuration
        output_dir: Directory to save validation outputs
        validation_tick: Tick to use for validation
        
    Returns:
        Dictionary with validation results and metrics
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validation_logger.info("Starting validation workflow")
    validation_logger.info(f"Parameter values: {parameter_values}")
    validation_logger.info(f"Configs: {config_files}")
    validation_logger.info(f"Runs per config: {runs_per_config}")
    
    validation_logger.info("Loading experimental data...")
    
    # This has the form of a DataFrame with columns: 
    # group, time_hour, (raw values), small_scaffold_cell_avg, small_scaffold_collagen_avg
    # we want to extract the small scaffold experimental data

    experimental_data = extract_small_scaffold_experimental(file_path=experimental_data_file)

    # Convert this into the expected dict format
    experimental_data_dict: dict[str, dict[int, dict[str, float]]] = convert_df_to_dict(experimental_data)
    
    validation_logger.info("Running validation simulations...")
    simulation_results = run_multiple_validations(
        parameter_values=parameter_values,
        subprocess_run_dir=subprocess_run_dir,
        bin_dir=bin_dir,
        config_file_dir=config_file_dir,
        config_files=config_files,
        runs_per_config=runs_per_config,
        tracked_ticks=[constants.TICKS_PER_DAY * 3, constants.TICKS_PER_DAY * 6, constants.TICKS_PER_DAY * 9]
    )
    
    validation_logger.info("Calculating validation metrics...")    

    metrics = calculate_validation_metrics(
        simulation_results=simulation_results,
        experimental_data=experimental_data_dict
    )
    
    # Return all results
    return {
        "experimental_data": experimental_data,
        "simulation_results": simulation_results,
        "metrics": metrics
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate ABM model with optimized parameters")
    parser.add_argument("--param-file", type=str, required=True, help="Path to file with optimized parameters")
    parser.add_argument("--run-dir", type=str, required=True, help="Directory for simulation runs")
    parser.add_argument("--bin-dir", type=str, required=True, help="Directory containing ABM binaries")
    parser.add_argument("--config-dir", type=str, required=True, help="Directory containing configuration files")
    parser.add_argument("--exp-data", type=str, required=True, help="Path to experimental data file")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per configuration")
    parser.add_argument("--output-dir", type=str, default="output/validation", help="Directory to save validation outputs")
    
    args = parser.parse_args()
    
    # Load optimized parameters
    with open(args.param_file, 'r') as f:
        parameters = json.load(f)
    
    # Run validation
    run_validation(
        parameter_values=parameters["best_parameters"],
        subprocess_run_dir=Path(args.run_dir),
        bin_dir=Path(args.bin_dir),
        config_file_dir=Path(args.config_dir),
        experimental_data_file=Path(args.exp_data),
        runs_per_config=args.runs,
        output_dir=Path(args.output_dir)
    )
