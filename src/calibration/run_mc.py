import click
import spotpy
import logging
import os
import json
import sys
from pathlib import Path
from src.calibration.error_function import normalized_biomarker_error
from src.calibration.setup_abm_spotpy import spotpyABM
from src.abm_world.preprocessing import process_parameters_from_csv, extract_n_parameters


def setup_main_logger(log_level=logging.INFO):
    """
    Set up a dedicated logger for the main process with both console and file output.
    
    Args:
        log_level: Logging level (default: INFO)
    
    Returns:
        The configured logger instance
    """
    main_logger = logging.getLogger('main_process')
    main_logger.setLevel(log_level)
    main_logger.propagate = False  # Don't propagate to root logger
    
    if main_logger.handlers:
        main_logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler('output/optimization_main.log', mode='w')
    
    console_handler.setLevel(log_level)
    file_handler.setLevel(log_level)
    
    formatter = logging.Formatter('%(asctime)s - MAIN - %(levelname)s - %(message)s')
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    main_logger.addHandler(console_handler)
    main_logger.addHandler(file_handler)
    
    return main_logger


# Initialize basic logging but we'll use our custom logger in the main function
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.command()
@click.option('--log-level', '-l', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], case_sensitive=False), 
              help="Set the logging level.")
@click.option('--param-ranking', '-pr', type=click.Choice(choices=['random_forest', 'morris'], case_sensitive=False), default='random_forest',
              help="Method to rank parameters for sensitivity analysis.")
@click.option('--param-num', '-pn', default=5, type=int, 
              help="Number of parameters to rank.")
@click.option('--num-iterations', '-i', default=800, type=int, 
              help="Number of iterations for SPOTPY optimization.")
@click.option('--run-dir-parent', '-r', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Parent directory for simulation run subdirectories.")
@click.option('--sensitivity-analysis-csv', '-s', type=click.Path(exists=True, dir_okay=False),
              help="CSV file containing sensitivity analysis results.")
@click.option('--experimental-data-csv', '-e', required=True, type=click.Path(exists=True, dir_okay=False),
             help="CSV file containing experimental/observed data for evaluation.")
@click.option('--config-file-dir', '-c', default=Path("configFiles"), type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Directory containing configuration files.")
@click.option('--bin-dir', '-b', default=Path("bin"), type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Directory containing the ABM simulation binary files.")
@click.option('--parallel', '-p', type=click.Choice(['mpc', 'mpi', 'seq'], case_sensitive=False), default='mpc',
              help="Parallelization method to use.")
@click.option('--save-runs', '-sr', type=bool, default=False,
              help="Flag to save individual simulation runs. If not set, only the final optimization results are saved.")
@click.pass_context
def run(ctx, 
        log_level: str,
        param_ranking: str, 
        param_num: int, 
        num_iterations: int, 
        run_dir_parent: Path,
        sensitivity_analysis_csv: Path,
        config_file_dir: Path,
        bin_dir: Path,
        experimental_data_csv: Path,
        parallel: str,
        save_runs: bool,
        ):
    """
    Command line interface for running the ABM simulation with spotpy.
    """
    # Set up the main process logger
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    main_logger = setup_main_logger(log_level=log_level_map.get(log_level.upper(), logging.INFO))
    main_logger.info(f"Starting ABM optimization with {param_num} parameters")
    main_logger.info(f"Using parallelization method: {parallel}")
    main_logger.info(f"Running for {num_iterations} iterations")
    run_dir_parent = Path(run_dir_parent)
    sensitivity_analysis_csv = Path(sensitivity_analysis_csv)
    config_file_dir = Path(config_file_dir)
    bin_dir = Path(bin_dir)
    experimental_data_csv = Path(experimental_data_csv)

    main_logger.info(f"Processing parameters from {sensitivity_analysis_csv}")
    df = process_parameters_from_csv(sensitivity_analysis_csv)
    chosen_params = extract_n_parameters(df, param_ranking, param_num)

    main_logger.info(f"Selected {len(chosen_params)} parameters using {param_ranking} ranking method")
    main_logger.info(f"Parameters to optimize: {chosen_params}")

    main_logger.info("Initializing spotpy setup")
    spotpy_setup = spotpyABM(
        error_function=normalized_biomarker_error,
        subprocess_run_dir=run_dir_parent,
        bin_dir=bin_dir,
        config_file_dir=config_file_dir,
        all_params=df,
        chosen_params=chosen_params,
        experimental_data_file=experimental_data_csv,
        num_ticks=289,
        tracked_ticks=[144, 288],
        save_runs=save_runs
    )

    main_logger.info(f"Creating Monte Carlo sampler with {parallel} parallelization")
    sampler = spotpy.algorithms.mc(
        spotpy_setup,
        parallel=parallel,
        dbname='mc_abm_optimization',
        dbformat='csv',
        save_sim=True if save_runs else False
    )
    
    main_logger.info(f"Starting sampling with {num_iterations} iterations")
    sampler.sample(
        repetitions=num_iterations
    )
    
    main_logger.info("Sampling completed, generating final report")
    
    # Generate final progress report
    final_report = spotpy_setup.generate_progress_report()
    
    with open('output/optimization_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    # Move the ROPE database file to the output directory
    import shutil
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    db_filename = 'mc_abm_optimization.csv'
    if os.path.exists(db_filename):
        main_logger.info(f"Moving database file {db_filename} to output directory")
        shutil.copy(db_filename, output_dir / db_filename)
        main_logger.info(f"Database saved to {output_dir / db_filename}")
    
    main_logger.info("Optimization completed. Check 'output/optimization_progress.log' for detailed progress tracking.")
    main_logger.info("Final report saved to 'output/optimization_report.json'.")
    main_logger.info("Main process log saved to 'output/optimization_main.log'.")

if __name__ == "__main__":
    run()