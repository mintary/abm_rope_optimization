import spotpy
import uuid
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Callable
from spotpy.objectivefunctions import rmse
import src.calibration.constants as constants
from src.abm_world.simulation import run_abm_simulation, prepare_sample
from src.abm_world.preprocessing import extract_small_scaffold_experimental

logger = logging.getLogger(__name__)

class spotpyABM(object):
    """
    A class to set up and run ABM simulations using the SpotPy framework.
    """
    def __init__(
            self, 
            subprocess_run_dir: Path,
            bin_dir: Path,
            config_file_paths: list[Path],
            chosen_params: list[str],
            all_params: pd.DataFrame,
            num_ticks: int = 289,
            tracked_biomarkers: list[str] = ["TotalFibroblast", "Collagen"],
            tracked_ticks: list[int] = [constants.TICKS_PER_DAY * 3, constants.TICKS_PER_DAY * 6],
            error_function: Callable[[list[float], list[float]], float] = rmse,
            experimental_data_file: Path = Path("input/experimental.csv"),
            save_runs: bool = False
            ):
        """
        Args:
            subprocess_run_dir (Path): Directory where all the subprocess runs will be executed.
            bin_dir (Path): Directory containing the ABM binaries.
            config_file_dir (Path): Directory containing the configuration files for the ABM.
            chosen_params (list[str]): List of parameter names to optimize.
            all_params (pd.DataFrame): DataFrame containing all parameters for the ABM.
            num_ticks (int): Total number of ticks for the simulation.
            tracked_biomarkers (list[str]): List of biomarkers to track during the simulation, returned as our results of the simulation.
            tracked_ticks (list[int]): Ticks at which to track biomarkers.
            error_function (Optional[Callable[[list[float], list[float]], float]]): Custom error function to evaluate the simulation results.
        """
        self.subprocess_run_dir = subprocess_run_dir
        self.bin_dir = bin_dir
        self.config_file_paths = config_file_paths
        self.chosen_params = chosen_params
        self.all_params = all_params
        self.params = []
        self.num_ticks = num_ticks
        self.tracked_biomarkers = tracked_biomarkers
        self.tracked_ticks = tracked_ticks
        self.error_function = error_function
        self.experimental_data_file = experimental_data_file
        self.save_runs = save_runs
        
        # Progress tracking
        self.iteration_count = 0
        self.best_error = float('inf')
        self.initial_error = None
        self.error_history = []
        self.subset_errors = {}  # Track best error per ROPE subset
        
        # Set up dedicated progress logger
        self.progress_logger = self._setup_progress_logger()
    
    def _setup_progress_logger(self) -> logging.Logger:
        """Set up a dedicated logger for progress reports"""
        progress_logger = logging.getLogger(f"{__name__}.progress")
        progress_logger.setLevel(logging.INFO)
        
        for handler in progress_logger.handlers[:]:
            progress_logger.removeHandler(handler)
        
        progress_file = Path("output/optimization_progress.log")
        file_handler = logging.FileHandler(progress_file, mode='w')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - PROGRESS - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        progress_logger.addHandler(file_handler)
        progress_logger.propagate = False  # Don't propagate to root logger
        
        return progress_logger

    def parameters(self) -> np.ndarray:
        """
        Generate parameter vector based on the chosen parameters.
        """
        filtered_params = self.all_params[self.all_params['parameter_name'].isin(self.chosen_params)]
        if filtered_params.empty:
            raise ValueError("No parameters found matching the chosen parameters.")
        
        spotpy_params = [
            spotpy.parameter.Uniform(
                name=str(row['parameter_name']),
                low=float(row['lower_bound']),
                high=float(row['upper_bound']),
                optguess=float(row['default_value'])
            ) for _, row in filtered_params.iterrows()
        ]

        return spotpy.parameter.generate(spotpy_params)

    def simulation(self, vector: np.ndarray) -> list[float]:
        """
        Run the ABM simulation with the provided parameters.
        Args:
            vector (np.ndarray): Sampled parameter values from SpotPy. These are in the order of the chosen parameters.
        Returns:
            list[float]: List of tracked biomarkers at the specified ticks for each config file, flattened into a single list.
            This has the form: 
            ```
            [gh2_collagen_72, gh2_cell_72, gh2_cell_144, gh2_collagen_144, 
            gh5_collagen_72, gh5_cell_72, gh5_cell_144, gh5_collagen_144, 
            gh10_collagen_72, gh10_cell_72, gh10_cell_144, gh10_collagen_144]
            ```
        """
        run_id = uuid.uuid4().hex 

        run_id_dir = self.subprocess_run_dir / run_id
        run_id_dir.mkdir(parents=True, exist_ok=True)

        # Copy the binary executable to the run directory
        run_id_dir_bin = run_id_dir / "bin"
        run_id_dir_bin.mkdir(parents=True, exist_ok=True)
        for item in self.bin_dir.iterdir():
            if item.is_file():
                dest = run_id_dir_bin / item.name
                shutil.copy(item, dest)

        config_dir = run_id_dir / "configFiles"
        config_dir.mkdir(parents=True, exist_ok=True)

        for config_file_path in self.config_file_paths:
            dest = config_dir / config_file_path.name
            shutil.copy(config_file_path, dest)

        sample_output_path = run_id_dir / "Sample.txt"

        # Pass all parameters to the prepare_sample function
        prepare_sample(vector, self.all_params, self.chosen_params, sample_output_path)

        results: list[float] = []

        # Run simulation for GH2, GH5, then GH10
        for config_file_path in self.config_file_paths:
            biomarker_output_dir = run_id_dir / "output"
            biomarker_output_dir.mkdir(parents=True, exist_ok=True)
            bin_stderr_path = biomarker_output_dir / "ABM_simulation_stderr.txt"
            bin_stdout_path = biomarker_output_dir / "ABM_simulation_stdout.txt"

            run_results = run_abm_simulation(
                bin_path=run_id_dir_bin,
                bin_stderr_path=bin_stderr_path,
                bin_stdout_path=bin_stdout_path,
                biomarker_output_dir=biomarker_output_dir,
                sample_path=sample_output_path,
                config_path=config_file_path,
                tracked_biomarkers=self.tracked_biomarkers,
                num_ticks=self.num_ticks,
                tracked_ticks=self.tracked_ticks,
                cwd=run_id_dir, # We set this to the cwd where the simulation is run
            )

            print(f"Simulation results for config {config_file_path.name}: {run_results}")

            # Flatten the result into a single list with the format:
            # [gh2_collagen_72h, gh2_cell_72h, gh2_collagen_144h, gh2_cell_144h, ...]
            for tick in self.tracked_ticks:
                if tick in run_results:
                    tick_result = run_results[tick]
                    flattened_result = [tick_result[biomarker] for biomarker in self.tracked_biomarkers]
                    results.extend(flattened_result)

        # Remove directory if save_runs is False
        if not self.save_runs:
            shutil.rmtree(run_id_dir)

        return results
    
    def evaluation(self) -> list[float]:
        """
        Returns the evaluation data as a flat list of biomarker values for each configuration and tracked time point.
        The format is:
        [<config1_biomarker1_time1>, <config1_biomarker2_time1>, ..., <config1_biomarkerN_timeM>, 
         <config2_biomarker1_time1>, ..., <configK_biomarkerN_timeM>]
        where each entry corresponds to a biomarker value for a specific configuration and time point.
        The order matches the configurations and tracked times specified in the setup.
        """
        df = extract_small_scaffold_experimental(self.experimental_data_file)
        df = df.sort_values(['group', 'time_hour']).reset_index(drop=True)
        
        evaluation_data: list[float] = []

        for group in self.config_file_paths:
            group_name = group.stem # Remove extension to get the group name
            group_data = df[df['group'] == group_name].sort_values('time_hour')
            
            group_data = group_data[(group_data['time_hour'] != 216) & (group_data['time_hour'] != 0)]
            
            # Extract the values for collagen and cells at 72h and 144h
            for time_hour in [72, 144]:
                if time_hour in group_data['time_hour'].values:
                    row = group_data[group_data['time_hour'] == time_hour].iloc[0] 
                    evaluation_data.append(row['small_scaffold_cell_avg'])
                    evaluation_data.append(row['small_scaffold_collagen_pg'])

        return evaluation_data

    def objectivefunction(self, simulation: list[float], evaluation: list[float], params=None) -> float:
        """
        Calculate the objective function value based on the simulation results and evaluation data.
        Args:
            simulation (list[float]): List of tracked biomarkers from the simulation.
            evaluation (list[float]): List of expected values from the experimental data.
            params: The parameter values used for this simulation. In SpotPy, this is a tuple of (params, parnames).
        Returns:
            float: The objective function value. Since ROPE maximizes by default, we return 
                   the negative error so that maximizing the objective minimizes the error.
        """        
        if len(simulation) != len(evaluation):
            raise ValueError("Simulation and evaluation lists must be of the same length.")

        error = self.error_function(simulation, evaluation)
        
        # Track progress for reporting (using the actual positive error)
        param_list = None
        if params is not None:
            # SpotPy passes params as a tuple of (param_values, param_names)
            # I wish they had type hints so bad
            if isinstance(params, tuple) and len(params) > 0:
                param_values = params[0]
                if hasattr(param_values, 'tolist'):
                    param_list = param_values.tolist()
                elif isinstance(param_values, list):
                    param_list = param_values
        
        self.track_progress(error, parameter_values=param_list)
        
        # Return negative error so ROPE algorithm minimizes error by maximizing -error
        return -error
    
    def track_progress(self, current_error: float, subset: int = 0, parameter_values: Optional[list] = None) -> None:
        """
        Track optimization progress and log percentage improvements.
        
        Args:
            current_error: The current error value
            subset: The current ROPE subset
            parameter_values: The parameter values for this iteration
        """
        self.iteration_count += 1
        self.error_history.append(current_error)
        
        if self.initial_error is None:
            # Print the experimental data 
            self.progress_logger.info(f"""
                                      {self.evaluation()}
                                      """)
            self.initial_error = current_error
            self.progress_logger.info(f"Initial error: {current_error:.6f}")
            if parameter_values:
                param_str = ", ".join([f"{p:.6f}" for p in parameter_values])
                self.progress_logger.info(f"Initial parameters: [{param_str}]")
            return
            
        # Calculate percentage from initial
        pct_from_initial = ((self.initial_error - current_error) / self.initial_error) * 100 if self.initial_error > 0 else 0
        
        # Format parameter values if provided
        param_info = ""
        if parameter_values:
            param_str = ", ".join([f"{p:.6f}" for p in parameter_values])
            param_info = f" Parameters: [{param_str}]"
        
        # Always log every iteration
        self.progress_logger.info(f"Iteration {self.iteration_count}: Error {current_error:.6f} "
                                  f"(Improvement from initial: {pct_from_initial:.2f}%){param_info}")
        
        # Additional logging for new best error
        if current_error < self.best_error:
            self.best_error = current_error
            
            # Calculate percentage improvement from previous best
            if len(self.error_history) > 1:
                prev_best = min(self.error_history[:-1])
                recent_improvement = ((prev_best - current_error) / prev_best) * 100 if prev_best > 0 else 0
                
                message = (f"NEW BEST at iteration {self.iteration_count}: Error {current_error:.6f} "
                          f"(Total improvement: {pct_from_initial:.2f}%, "
                          f"Recent improvement: {recent_improvement:.2f}%)")
                self.progress_logger.info(message)
        
        # Track subset progress for ROPE
        if subset not in self.subset_errors:
            self.subset_errors[subset] = []
        self.subset_errors[subset].append(current_error)
        
    def log_subset_summary(self, subset: int) -> None:
        """
        Log summary statistics for a completed ROPE subset.
        """
        if subset in self.subset_errors:
            subset_best = min(self.subset_errors[subset])
            subset_worst = max(self.subset_errors[subset])
            subset_avg = np.mean(self.subset_errors[subset])
            
            if self.initial_error:
                improvement = ((self.initial_error - subset_best) / self.initial_error) * 100
                message = (f"SUBSET COMPLETE: Subset {subset}: Best={subset_best:.6f}, "
                          f"Avg={subset_avg:.6f}, Worst={subset_worst:.6f}, "
                          f"Total improvement={improvement:.2f}%")
                self.progress_logger.info(message)
    
    def generate_progress_report(self) -> dict:
        """
        Generate a comprehensive progress report.
        """
        if not self.error_history:
            return {}
            
        report = {
            'total_iterations': len(self.error_history),
            'initial_error': self.initial_error,
            'final_error': self.best_error,
            'total_improvement_pct': ((self.initial_error - self.best_error) / self.initial_error) * 100 if self.initial_error else 0,
            'error_change_pct': ((self.best_error - self.initial_error) / self.initial_error) * 100 if self.initial_error else 0,
            'convergence_rate': self._calculate_convergence_rate(),
            'subset_summary': self._get_subset_summary()
        }
        
        # Also log to progress file
        self.progress_logger.info("="*50)
        self.progress_logger.info("FINAL OPTIMIZATION REPORT")
        self.progress_logger.info("="*50)
        self.progress_logger.info(f"Total iterations: {report['total_iterations']}")
        self.progress_logger.info(f"Initial error: {report['initial_error']:.6f}")
        self.progress_logger.info(f"Best final error: {report['final_error']:.6f}")
        self.progress_logger.info(f"Total improvement: {report['total_improvement_pct']:.2f}%")
        self.progress_logger.info(f"Error change: {report['error_change_pct']:.2f}%")
        self.progress_logger.info(f"Convergence rate: {report['convergence_rate']:.4f}")
        
        # Log subset summaries
        if report['subset_summary']:
            self.progress_logger.info("\nSubset Performance Summary:")
            for subset_id, summary in report['subset_summary'].items():
                self.progress_logger.info(f"  Subset {subset_id}: Best={summary['best']:.6f}, "
                                        f"Avg={summary['avg']:.6f}, Improvement={summary['improvement']:.2f}%")
        
        self.progress_logger.info("="*50)
        
        return report
    
    def _calculate_convergence_rate(self) -> float:
        """
        Calculate the rate of convergence (exponential decay constant).
        """
        if len(self.error_history) < 10:
            return 0.0
            
        # Use last 50% of iterations to estimate convergence
        mid_point = len(self.error_history) // 2
        recent_errors = self.error_history[mid_point:]
        
        # Simple convergence rate based on error reduction
        if len(recent_errors) > 1:
            initial_recent = recent_errors[0]
            final_recent = recent_errors[-1]
            if initial_recent > 0:
                return (initial_recent - final_recent) / (initial_recent * len(recent_errors))
        return 0.0
    
    def _get_subset_summary(self) -> dict:
        """
        Get summary of each ROPE subset performance.
        """
        summary = {}
        for subset, errors in self.subset_errors.items():
            if errors:
                summary[subset] = {
                    'best': min(errors),
                    'worst': max(errors),
                    'avg': np.mean(errors),
                    'improvement': ((self.initial_error - min(errors)) / self.initial_error) * 100 if self.initial_error else 0
                }
        return summary
