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
@click.option('--run-dir-parent', '-r', default=Path("output/rope_runs"), type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Parent directory for simulation run subdirectories.")
@click.option('--sensitivity-analysis-csv', '-s', default=Path("input/sensitivity_analysis.csv"), type=click.Path(exists=True, dir_okay=False),
              help="CSV file containing sensitivity analysis results.")
@click.option('--experimental-data-csv', '-e', default=Path("input/experimental.csv"), type=click.Path(exists=True, dir_okay=False),
              help="CSV file containing experimental/observed data for evaluation.")
@click.option('--config-files', '-cn', default=["config_Scaffold"], type=list[click.Path(exists=True, file_okay=True, dir_okay=False)],
              help="Comma-separated list of configuration files for the ABM simulation.")
@click.option('--bin-dir', '-b', default=Path("bin"), type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Directory containing the ABM simulation binary files.")
@click.option('--parallel', '-p', type=click.Choice(['mpc', 'mpi', 'seq'], case_sensitive=False), default='mpc',
              help="Parallelization method to use.")
@click.option('--save-runs', '-sr', type=bool, default=False,
              help="Flag to save individual simulation runs. If not set, only the final optimization results are saved.")
@click.option('--repetitions-first-run', '-rf', default=0, type=int,
              help="Number of repetitions for the first run.")
@click.option('--subsets', '-sbs', default=6, type=int,
              help="Number of subsets for the ROPE sampler.")
@click.option('--percentage-first-run', '-pfr', default=0.1, type=float,
              help="Percentage of the first run to use for the ROPE sampler.")
@click.option('--percentage-following-runs', '-pfrs', default=0.1, type=float,
              help="Percentage of the following runs to use for the ROPE sampler.")
@click.pass_context
def run(ctx, 
        log_level: str,
        param_ranking: str, 
        param_num: int, 
        num_iterations: int, 
        run_dir_parent: Path,
        sensitivity_analysis_csv: Path,
        config_file_paths: list[Path],
        bin_dir: Path,
        experimental_data_csv: Path,
        parallel: str,
        save_runs: bool,
        repetitions_first_run: int,
        subsets: int,
        percentage_first_run: float,
        percentage_following_runs: float
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
    config_file_paths = [Path(p) for p in config_file_paths]
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

    main_logger.info(f"Creating ROPE sampler with {parallel} parallelization")
    sampler = spotpy.algorithms.rope(
        spotpy_setup,
        dbname='rope_abm_optimization',
        dbformat='csv',
        parallel=parallel,
        save_sim=True,
    )
    
    main_logger.info(f"Starting sampling with {num_iterations} iterations")
    sampler.sample(
        repetitions=num_iterations,
        repetitions_first_run=repetitions_first_run if repetitions_first_run > 0 else num_iterations // 2,
        subsets=subsets,
        percentage_first_run=percentage_first_run,
        percentage_following_runs=percentage_following_runs
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
    
    db_filename = 'rope_abm_optimization.csv'
    if os.path.exists(db_filename):
        main_logger.info(f"Moving database file {db_filename} to output directory")
        shutil.copy(db_filename, output_dir / db_filename)
        main_logger.info(f"Database saved to {output_dir / db_filename}")
    
    main_logger.info("Optimization completed. Check 'output/optimization_progress.log' for detailed progress tracking.")
    main_logger.info("Final report saved to 'output/optimization_report.json'.")
    main_logger.info("Main process log saved to 'output/optimization_main.log'.")

if __name__ == "__main__":
    run()