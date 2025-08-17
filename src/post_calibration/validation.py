"""
Validation module for ABM model with optimized parameters.

This module provides functionality to validate the ABM model by:
1. Running simulations with optimized parameters
2. Comparing results to validation data (day 9, time_hour=216)
3. Running multiple simulations per configuration to account for stochasticity
4. Calculating validation metrics
"""

import json
import shutil
import logging
import numpy as np
import pandas as pd
import sys
import click
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Mapping
from scipy import stats
from src.abm_world.simulation import prepare_sample, run_abm_simulation
from src.abm_world.preprocessing import extract_small_scaffold_experimental, process_parameters_from_csv, extract_n_parameters
from src.calibration.error_function import normalized_biomarker_error
from src.post_calibration.parameter_extraction import extract_parameters_from_csv
import src.calibration.constants as constants

logger = logging.getLogger(__name__)

def setup_main_logger(log_level=logging.INFO):
    """
    Set up a dedicated logger for the main process with both console and file output.
    
    Args:
        log_level: Logging level (default: INFO)
    
    Returns:
        The configured logger instance
    """
    main_logger = logging.getLogger('validation_main')
    main_logger.setLevel(log_level)
    main_logger.propagate = False  # Don't propagate to root logger
    
    if main_logger.handlers:
        main_logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create output directory if it doesn't exist
    output_dir = Path("output/validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / 'validation_main.log', mode='w')
    
    console_handler.setLevel(log_level)
    file_handler.setLevel(log_level)
    
    formatter = logging.Formatter('%(asctime)s - VALIDATION_MAIN - %(levelname)s - %(message)s')
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    main_logger.addHandler(console_handler)
    main_logger.addHandler(file_handler)
    
    return main_logger

validation_logger = logging.getLogger(f"{__name__}.validation")
validation_logger.setLevel(logging.DEBUG)

if not validation_logger.handlers:
    log_dir = Path("output/validation")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / "validation.log", mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - VALIDATION - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    validation_logger.addHandler(file_handler)
    validation_logger.propagate = False  # Don't propagate to root logger


def run_multiple_validations(
        parameter_values: List[float],
        all_params: pd.DataFrame,
        chosen_params: List[str],
        subprocess_run_dir: Path,
        bin_dir: Path,
        config_file_dir: Path,
        config_files: List[str],
        runs_per_config: int = 3,
        num_ticks: int = 500,
        tracked_biomarkers: List[str] = ["TotalFibroblast", "Collagen"],
        tracked_ticks: List[int] = [constants.TICKS_PER_DAY * 3, constants.TICKS_PER_DAY * 6, constants.TICKS_PER_DAY * 9],
) -> Dict[str, Dict[int, Dict[str, List[float]]]]:
    """
    Run multiple validation simulations for each configuration to account for stochasticity.
    
    Args:
        parameter_values: Optimized parameter values to use
        all_params: DataFrame containing all parameter values
        chosen_params: List of parameter names matching the parameter_values
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
        config_base = Path(config_file_name).name  # get just the filename
        config_name = config_base.replace("config_", "").replace(".txt", "")
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
            dest_config = run_config_dir / config_base
            shutil.copy(source_config, dest_config)

            validation_logger.info(f"  Preparing sample file for run {run_idx} in {run_dir}")

            sample_path = run_dir / "Sample.txt"

            prepare_sample(parameter_values, all_params, chosen_params, sample_path)

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
    validation_logger.debug(f"Simulation results: {simulation_results}")
    validation_logger.debug(f"Experimental data: {experimental_data}")
    for config_name, config_results in simulation_results.items():
        if config_name not in metrics:
            metrics[config_name] = {}

        if simulation_validation_tick not in config_results:
            validation_logger.warning(f"Validation tick {simulation_validation_tick} not found in results for {config_name}")
            continue

        # Map simulation config names to experimental config names
        # Simulation: "Scaffold_GH2" -> Experimental: "GH2"
        exp_config_name = config_name.replace("Scaffold_", "")
        
        validation_logger.debug(f"Mapping simulation config '{config_name}' to experimental config '{exp_config_name}'")
        
        if exp_config_name not in experimental_data or experimental_validation_hour not in experimental_data[exp_config_name]:
            validation_logger.warning(f"No experimental data for {exp_config_name} at hour {experimental_validation_hour}")
            continue

        sim_data = config_results[simulation_validation_tick]
        exp_data = experimental_data[exp_config_name][experimental_validation_hour]

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
            
            # Handle case where std is 0 (all values identical)
            if std_value == 0 or sem == 0:
                # When std is 0, CI is just the mean value
                ci = (mean_value, mean_value)
            else:
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
        # Strip 'config_Scaffold_' prefix from group name to match expected keys (e.g., 'GH2')
        group = row['group']
        if isinstance(group, str) and group.startswith('config_Scaffold_'):
            group = group.replace('config_Scaffold_', '')
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
        all_params: pd.DataFrame,
        chosen_params: List[str],
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
        all_params=all_params,
        chosen_params=chosen_params,
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

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

@click.command()
@click.option('--param-file', '-p', type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to file with optimized parameters (JSON or CSV)")
@click.option('--run-dir', '-r', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True,
              help="Directory for simulation runs")
@click.option('--bin-dir', '-b', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True,
              help="Directory containing ABM binaries")
@click.option('--config-dir', '-c', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True,
              help="Directory containing configuration files")
@click.option('--sensitivity-analysis-csv', '-s', type=click.Path(exists=True, dir_okay=False),
              help="CSV file containing sensitivity analysis results.")
@click.option('--param-ranking', '-pr', type=click.Choice(choices=['random_forest', 'morris'], case_sensitive=False), default='random_forest',
              help="Method to rank parameters for sensitivity analysis.")
@click.option('--param-num', '-pn', default=5, type=int, 
              help="Number of parameters to rank.")
@click.option('--exp-data', '-e', type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to experimental data file")
@click.option('--runs', type=int, default=3,
              help="Number of runs per configuration (default: 3)")
@click.option('--output-dir', '-o', type=click.Path(file_okay=False, dir_okay=True), 
              default="output/validation", help="Directory to save validation outputs")
@click.option('--config-file', '-cf', multiple=True, default=["config_Scaffold_GH2.txt", "config_Scaffold_GH5.txt", "config_Scaffold_GH10.txt"],
              help="Configuration file to use (specify once per file)")
@click.option('--use-csv', is_flag=True, 
              help="Load parameters from CSV file instead of JSON")
@click.option('--num-params', type=int, default=5,
              help="Number of parameters in the CSV file (only used with --use-csv)")
@click.option('--log-level', '-ll', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], case_sensitive=False), 
              help="Set the logging level.")
def validate(
    param_file, 
    run_dir, 
    bin_dir, 
    config_dir, 
    sensitivity_analysis_csv, 
    param_ranking, 
    param_num,
    exp_data, 
    runs, 
    output_dir, 
    config_file, 
    use_csv, 
    num_params,
    log_level):
    """
    Validate ABM model with optimized parameters.
    
    This command runs validation simulations using optimized parameters from either
    a JSON file (from parameter extraction) or directly from a CSV file (ROPE output).
    """
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    main_logger = setup_main_logger(log_level=log_level_map.get(log_level.upper(), logging.INFO))
    
    main_logger.info("Starting ABM model validation...")
    main_logger.info(f"Parameter file: {param_file}")
    main_logger.info(f"Using {'CSV' if use_csv else 'JSON'} format")
    main_logger.info(f"Runs per configuration: {runs}")
    
    if use_csv:        
        main_logger.info("Loading parameters from CSV file...")
        param_data = extract_parameters_from_csv(param_file, num_parameters=num_params, top_n=1)
        parameter_values = param_data["best_parameter_set"]
        
        main_logger.info(f"Best parameter set: {parameter_values}")
        main_logger.info(f"Error value: {param_data['actual_error_value']:.6f}")
    else:
        main_logger.info("Loading parameters from JSON file...")
        with open(param_file, 'r') as f:
            parameters = json.load(f)
        
        if "best_parameter_set" in parameters:
            parameter_values = parameters["best_parameter_set"]
        elif "best_parameters" in parameters:
            parameter_values = parameters["best_parameters"][0]
        else:
            raise ValueError("Unrecognized parameter file format. Expected 'best_parameter_set' or 'best_parameters' key.")
    
    main_logger.info(f"Configuration files: {list(config_file)}")

    # Get all the parameter values
    all_params = process_parameters_from_csv(sensitivity_analysis_csv)
    chosen_params = extract_n_parameters(all_params, ranking_method=param_ranking, n=param_num)
    
    main_logger.info(f"Using parameter values: {parameter_values}")

    validation_results = run_validation(
        parameter_values=parameter_values,
        all_params=all_params,
        chosen_params=chosen_params,
        subprocess_run_dir=Path(run_dir),
        bin_dir=Path(bin_dir),
        config_file_dir=Path(config_dir),
        experimental_data_file=Path(exp_data),
        config_files=list(config_file),
        runs_per_config=runs,
        output_dir=Path(output_dir)
    )
    
    main_logger.info("\n" + "="*50)
    main_logger.info("VALIDATION SUMMARY")
    main_logger.info("="*50)
    
    metrics = validation_results["metrics"]
    # Save metrics to output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_serializable_metrics = convert_numpy_types(metrics)
    
    with open(output_dir / "validation_metrics.json", 'w') as f:
        json.dump(json_serializable_metrics, f, indent=2)
    
    main_logger.info(f"Validation results saved to: {output_dir / 'validation_results.json'}")
    main_logger.info(f"Metrics saved to: {output_dir / 'validation_metrics.json'}")

    for config_name, config_metrics in metrics.items():
        main_logger.info(f"Configuration: {config_name}")
        for biomarker, biomarker_metrics in config_metrics.items():
            mean_val = biomarker_metrics['mean']
            exp_val = biomarker_metrics['experimental']
            rel_error = biomarker_metrics['relative_error'] * 100
            within_ci = biomarker_metrics['within_ci']
            
            main_logger.info(f"  {biomarker}:")
            main_logger.info(f"    Simulated (mean): {mean_val:.2f}")
            main_logger.info(f"    Experimental: {exp_val:.2f}")
            main_logger.info(f"    Relative error: {rel_error:.1f}%")
            main_logger.info(f"    Within 95% CI: {'YES' if within_ci else 'NO'}")
    
    main_logger.info(f"Detailed results saved to: {output_dir}")
    main_logger.info("Main process log saved to 'output/validation/validation_main.log'.")
    main_logger.info("Validation completed successfully!")


if __name__ == "__main__":
    validate()
