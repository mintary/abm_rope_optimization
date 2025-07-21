import numpy as np
from typing import List, Tuple, Optional, Callable
from spotpy.objectivefunctions import rmse

def normalized_biomarker_error(
        simulation: List[float],
        evaluation: List[float],
        metric_function: Callable = rmse,
        fibroblast_indices: Optional[List[int]] = None,
        collagen_indices: Optional[List[int]] = None,
        debug: bool = True
    ) -> float:
    """
    Calculate the error between simulation and evaluation data with normalization
    for different biomarker types (fibroblast counts and collagen levels).
    
    Args:
        simulation: Simulated values in the form [gh2_cell_72h, gh2_collagen_72h, gh2_cell_144h, gh2_collagen_144h, ...]
        evaluation: Experimental values in the same format as simulation
        metric_function: The function to calculate error (default: RMSE)
        fibroblast_indices: List of indices for fibroblast count values
        collagen_indices: List of indices for collagen level values
        debug: If True, print debug information
    Returns:
        Normalized error value
    """
    if len(simulation) != len(evaluation):
        raise ValueError("Simulation and evaluation lists must have the same length")
    
    noise = np.random.uniform(-1e-12, 1e-12) # We use this to prevent ties in likes, otherwise ROPE will complain

    # Get every other index for fibroblasts and collagen
    if fibroblast_indices is None:
        fibroblast_indices = list(range(0, len(simulation), 2))
    
    if collagen_indices is None:
        collagen_indices = list(range(1, len(simulation), 2))
    
    sim_fibroblasts = [simulation[i] for i in fibroblast_indices]
    eval_fibroblasts = [evaluation[i] for i in fibroblast_indices]
    
    sim_collagen = [simulation[i] for i in collagen_indices]
    eval_collagen = [evaluation[i] for i in collagen_indices]
    
    fibroblast_norm = np.mean(eval_fibroblasts) if eval_fibroblasts else 1.0
    collagen_norm = np.mean(eval_collagen) if eval_collagen else 1.0
    
    sim_fibroblasts_norm = [val / fibroblast_norm for val in sim_fibroblasts]
    eval_fibroblasts_norm = [val / fibroblast_norm for val in eval_fibroblasts]
    
    sim_collagen_norm = [val / collagen_norm for val in sim_collagen]
    eval_collagen_norm = [val / collagen_norm for val in eval_collagen]
    
    fibroblast_error = metric_function(sim_fibroblasts_norm, eval_fibroblasts_norm)
    collagen_error = metric_function(sim_collagen_norm, eval_collagen_norm)

    if debug:
        print(f"Simulation Fibroblasts: {sim_fibroblasts}")
        print(f"Simulation Collagen: {sim_collagen}")
        print(f"Evaluation Fibroblasts: {eval_fibroblasts}")
        print(f"Evaluation Collagen: {eval_collagen}")
        print(f"Fibroblast Error: {fibroblast_error}")
        print(f"Collagen Error: {collagen_error}")
        print(f"Normalized Fibroblasts: {sim_fibroblasts_norm} vs {eval_fibroblasts_norm}")
        print(f"Normalized Collagen: {sim_collagen_norm} vs {eval_collagen_norm}")
    
    return (fibroblast_error + collagen_error) / 2 + noise