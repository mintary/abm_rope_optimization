from pathlib import Path
from typing import Optional
import pandas as pd
import csv
import subprocess
import platform
import os
import logging
import numpy as np
import src.calibration.constants as constants

logger = logging.getLogger(__name__)

def prepare_sample(
    values,
    all_params: pd.DataFrame,
    chosen_params: list[str],
    sample_output_path: Path
) -> None:
    """
    Prepare a sample file for the ABM simulation.
    
    Args:
        values: Parameter values from SpotPy (could be ndarray or parameter object)
        all_params: DataFrame containing all parameters with their bounds and default values
        chosen_params: List of parameter names to include in the sample
        sample_output_path: Path where the sample file will be written
    """
    # Convert any iterable obejct into a numpy array
    if hasattr(values, '__iter__') and not isinstance(values, np.ndarray):
        values = np.array(list(values))
    elif not isinstance(values, np.ndarray):
        values = np.array(values)
    
    # Now we go through all the parameters, changing the values of the chosen parameters only
    # to the values from the SpotPy sample
    sample_values: list[float] = [0.0] * len(all_params)

    chosen_parameter_index = 0
    param_names = all_params['parameter_name'].values
    default_values = [float(value) for value in all_params['default_value'].values]

    for i, param_name in enumerate(param_names):
        if param_name in chosen_params:
            chosen_param_value = values[chosen_parameter_index]
            sample_values[i] = chosen_param_value
            chosen_parameter_index += 1
        else:
            sample_values[i] = default_values[i]

    with sample_output_path.open('w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(sample_values)

    logger.debug(f"Sample file written to {sample_output_path} with values: {sample_values}")

def run_abm_simulation(
        bin_path: Path,
        bin_stderr_path: Path,
        bin_stdout_path: Path,
        sample_path: Path,
        config_path: Path,
        biomarker_output_dir: Path,
        tracked_biomarkers: list[str],
        cwd: Optional[Path] = None,
        num_ticks: int = 289,
        tracked_ticks: list[int] = [constants.TICKS_PER_DAY * 3, constants.TICKS_PER_DAY * 6],
) -> dict[int, dict[str, float]]:
    """
    Run the ABM simulation with the provided parameters.
    Args:
        bin_path (Path): Path to the ABM binary.
        bin_stderr_path (Path): Path to the binary's stderr output.
        bin_stdout_path (Path): Path to the binary's stdout output.
        biomarker_output_dir (Path): Directory to save biomarker outputs.
        sample_input_path (Path): Path to the input file with sampled parameters.
        config_path (Path): Path to the configuration file for the simulation.
        cwd (Optional[Path]): Working directory to set for the simulation - files are written to based off the CWD.
        num_ticks (int): Total number of ticks for the simulation.
        tracked_ticks (list[int]): Ticks at which to track biomarkers.
        tracked_biomarkers (Optional[list[str]]): List of biomarkers to track. If None, all biomarkers are tracked.
    Returns:
        dict[int, dict[str, float]]: Dictionary containing the simulation results for the tracked biomarkers at each tracked tick.
        {
            144: {"TotalFibroblast": value1, "Collagen": value2, ...},
            288: {"TotalFibroblast": value1, "Collagen": value2, ...},
            ...
        }
    """
    # Ensure the sample was created and exists
    if not sample_path.exists():
        logger.error(f"Sample file not found: {sample_path}")
        raise FileNotFoundError(f"Sample file not found: {sample_path}")
    
    # Determine the executable name based on the platform
    if platform.system() == "Windows":
        exe_name = "testRun.exe"
    else:
        exe_name = "testRun"
    test_run_path = bin_path / exe_name

    logger.info(f"Running ABM simulation with executable: {test_run_path}")

    if not test_run_path.exists():
        logger.error(f"ABM binary not found: {test_run_path}")
        raise FileNotFoundError(f"ABM binary not found: {test_run_path}")
    
    logger.info(f"Running ABM simulation with binary: {test_run_path}")
    results: dict[int, dict[str, float]] = {}

    biomarker_output_dir.mkdir(parents=True, exist_ok=True)
    bin_stderr_path.parent.mkdir(parents=True, exist_ok=True)
    bin_stdout_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the executable path for the subprocess command
    if cwd:
        exe_path = str((Path(cwd).resolve() / "bin" / exe_name))
    else:
        exe_path = os.fspath(test_run_path)

    command = [
        exe_path,
        "--numticks", str(num_ticks),
        "--inputfile", str(config_path),
        "--wxw", "0.6",
        "--wyw", "0.6",
        "--wzw", "0.6"
    ]

    subprocess.run(
        command,
        stdout=bin_stdout_path.open('w'),
        stderr=bin_stderr_path.open('w'),
        cwd=str(Path(cwd).resolve()) if cwd else None
    )

    output_biomarkers_path = biomarker_output_dir / "Output_Biomarkers.csv"
    logger.debug(f"Output biomarkers will be saved to: {output_biomarkers_path}")

    if not output_biomarkers_path.exists():
        logger.error(f"Output biomarkers file not found: {output_biomarkers_path}")
        logger.error(f"Directory contents: {list(biomarker_output_dir.glob('*'))}")
        raise FileNotFoundError(f"Output biomarkers file not found: {output_biomarkers_path}")
    
    with output_biomarkers_path.open('r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        if header is None:
            logger.error(f"CSV file {output_biomarkers_path} is empty or malformed (no header found).")
            raise ValueError(f"CSV file {output_biomarkers_path} is empty or malformed (no header found).")
        for row in reader:
            if int(row["clock"]) in tracked_ticks:
                tick_result = {biomarker: float(row[biomarker]) for biomarker in header if biomarker != "clock"}

                # If tracking fibroblasts, sum the values across all fibroblast types
                if "TotalFibroblast" in tracked_biomarkers:
                    fibroblast_types = ["ActivatedFibroblast", "Fibroblast"]
                    tick_result["TotalFibroblast"] = sum(tick_result[ft] for ft in fibroblast_types)
                    # Remove the individual fibroblast types if we are tracking TotalFibroblast
                    for ft in fibroblast_types:
                        tick_result.pop(ft)

                # Remove all biomarkers not in the tracked list
                if tracked_biomarkers:
                    tick_result = {k: v for k, v in tick_result.items() if k in tracked_biomarkers}

                results[int(row["clock"])] = tick_result
    return results

        