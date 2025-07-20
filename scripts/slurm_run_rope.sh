#!/bin/bash
#SBATCH --account=def-nicoleli
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=64000M
#SBATCH --mail-user=${EMAIL}
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Create directory for storing slurm_logs
mkdir -p slurm_logs

module load StdEnv/2020 gcc/9.3.0 cuda/11.0 python/3.11 || { echo "Module load failed"; exit 1; }

# PACKAGE DIRECTORY AT FAILURE OR SUCCESS
current_date=$(date +"%Y-%m-%d_%H-%M-%S")
tarball_name="param_opt_${current_date}_rope.tar.gz"

# Create directory for storing tarballs if it doesn't exist
mkdir -p finished_runs
echo "The package directory will be saved to the finished_runs directory"


# trap to package directory on any exit (success or failure) excluding env
function package_dir()
{
  echo "Packaging directory (trap)..."
  cd "$SLURM_TMPDIR"
  tar --exclude="env" --exclude="./env" --exclude="$SLURM_SUBMIT_DIR/finished_runs" -czf "$SLURM_SUBMIT_DIR/finished_runs/$tarball_name" 
  echo "Packaged directory into: $SLURM_SUBMIT_DIR/finished_runs/$tarball_name"
  exit
}

trap 'package_dir' EXIT


echo "SLURM_TMPDIR: $SLURM_TMPDIR"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# Copy required files to SLURM_TMPDIR
echo "Copying files to SLURM_TMPDIR..."
cp -r "$SLURM_SUBMIT_DIR"/* "$SLURM_TMPDIR/" || { echo "Failed to copy files"; exit 1; }
cd "$SLURM_TMPDIR" || { echo "Failed to change to SLURM_TMPDIR"; exit 1; }

# Create output directory if it doesn't exist
mkdir -p output

echo "Files in working directory:"
ls -la

# Set up environment
python --version

echo "Creating virtual environment..."
python -m venv "$SLURM_TMPDIR/env"
if [ $? -ne 0 ]; then
    echo "Virtual environment creation failed."
    exit 1
fi

echo "Virtual environment created successfully."

if [ -f "$SLURM_TMPDIR/env/bin/activate" ]; then
    source "$SLURM_TMPDIR/env/bin/activate"
    echo "Virtual environment activated."
else
    echo "Failed to activate virtual env: $SLURM_TMPDIR/env/bin/activate not found."
    exit 1
fi

# Ensure we're using the virtual environment
export PYTHONPATH=""
export PATH="$SLURM_TMPDIR/env/bin:$PATH"

echo "Installing dependencies..."
pip install --upgrade pip || { echo "Failed to upgrade pip"; exit 1; } 

# Install packages
echo "Installing dependencies from list..."
pip install --no-index pandas click numpy pathos spotpy scikit-learn || { echo "Failed to install Python dependencies"; exit 1; }

pip freeze > installed_requirements.txt

export OMP_NUM_THREADS=32
export OMP_NESTED=TRUE

# Set script directory and project root (now relative to SLURM_TMPDIR)
SCRIPT_DIR="$SLURM_TMPDIR"
PROJECT_ROOT="$SLURM_TMPDIR"

echo "Project root: $PROJECT_ROOT"
echo "Running ROPE optimization with mock simulation..."

# Create output directory for simulation runs
RUN_DIR="$PROJECT_ROOT/output/rope_runs"
mkdir -p "$RUN_DIR"

# Set up paths for the CLI arguments (all relative to PROJECT_ROOT now)
SENSITIVITY_CSV="$PROJECT_ROOT/input/sensitivity_analysis.csv"
EXPERIMENTAL_CSV="$PROJECT_ROOT/input/experimental.csv"
CONFIG_DIR="$PROJECT_ROOT/input/configFiles"
BIN_DIR="$PROJECT_ROOT/VFB_ABM/bin"

# Validate required files exist
if [ ! -f "$SENSITIVITY_CSV" ]; then
    echo "Error: Sensitivity analysis CSV not found at $SENSITIVITY_CSV"
    echo "Available files in input/:"
    ls -la "$PROJECT_ROOT/input/" || echo "input/ directory not found"
    exit 1
fi

if [ ! -f "$EXPERIMENTAL_CSV" ]; then
    echo "Error: Experimental data CSV not found at $EXPERIMENTAL_CSV"
    exit 1
fi

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config files directory not found at $CONFIG_DIR"
    exit 1
fi

if [ ! -d "$BIN_DIR" ]; then
    echo "Error: Binary directory not found at $BIN_DIR"
    exit 1
fi

LOG_LEVEL="DEBUG"
PARAM_RANKING="random_forest"
PARAM_NUM=5
NUM_ITERATIONS=200
PARALLEL="mpc"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --log-level|-l)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --param-ranking|-pr)
            PARAM_RANKING="$2"
            shift 2
            ;;
        --param-num|-pn)
            PARAM_NUM="$2"
            shift 2
            ;;
        --num-iterations|-i)
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        --parallel|-p)
            PARALLEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -l, --log-level       Set logging level (DEBUG|INFO|WARNING|ERROR|CRITICAL)"
            echo "  -pr, --param-ranking  Parameter ranking method (random_forest|morris)"
            echo "  -pn, --param-num      Number of parameters to rank (default: 5)"
            echo "  -i, --num-iterations  Number of optimization iterations (default: 200)"
            echo "  -p, --parallel        Parallelization method (mpc|mpi|seq)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Log Level: $LOG_LEVEL"
echo "  Parameter Ranking: $PARAM_RANKING"
echo "  Number of Parameters: $PARAM_NUM"
echo "  Iterations: $NUM_ITERATIONS"
echo "  Parallelization: $PARALLEL"
echo "  Run Directory: $RUN_DIR"
echo "  Binary Directory: $BIN_DIR"
echo ""

cd "$PROJECT_ROOT"

# Function to copy output files to submission directory
copy_output_to_submission() {
    echo "Copying output files to submission directory..."
    cp -r output/* "$SLURM_SUBMIT_DIR/output/" 2>/dev/null || echo "No output files to copy yet"
    # Also copy log files if they exist
    cp *.log "$SLURM_SUBMIT_DIR/output/" 2>/dev/null || echo "No log files to copy"
    cp installed_requirements.txt "$SLURM_SUBMIT_DIR/output/" 2>/dev/null || echo "No requirements file to copy"
}

echo "Starting ROPE optimization..."
python -m src.calibration.run_rope \
    --log-level "$LOG_LEVEL" \
    --param-ranking "$PARAM_RANKING" \
    --param-num "$PARAM_NUM" \
    --num-iterations "$NUM_ITERATIONS" \
    --run-dir-parent "$RUN_DIR" \
    --sensitivity-analysis-csv "$SENSITIVITY_CSV" \
    --experimental-data-csv "$EXPERIMENTAL_CSV" \
    --config-file-dir "$CONFIG_DIR" \
    --bin-dir "$BIN_DIR" \
    --parallel "$PARALLEL" > output/output.txt 2>&1

ROPE_EXIT_CODE=$?

# Copy output files after ROPE optimization
copy_output_to_submission

if [ $ROPE_EXIT_CODE -eq 0 ]; then
    echo "ROPE optimization completed successfully, running parameter extraction..."
    python -m src.post_calibration.parameter_extraction >> output/output.txt 2>&1
    
    # Copy output files after parameter extraction
    copy_output_to_submission
    
    echo "Running validation..."
    python -m src.post_calibration.validation \
        --param-file "$PROJECT_ROOT/output/rope_abm_optimization.csv" \
        --run-dir "$RUN_DIR" \
        --bin-dir "$BIN_DIR" \
        --config-dir "$CONFIG_DIR" \
        --sensitivity-analysis-csv "$SENSITIVITY_CSV" \
        --param-ranking "$PARAM_RANKING" \
        --param-num "$PARAM_NUM" \
        --exp-data "$EXPERIMENTAL_CSV" \
        --use-csv >> output/output.txt 2>&1

    FINAL_EXIT_CODE=$?
    
    # Copy final output files after validation
    copy_output_to_submission
else
    echo "ROPE optimization failed with exit code: $ROPE_EXIT_CODE"
    FINAL_EXIT_CODE=$ROPE_EXIT_CODE
fi

if [ $FINAL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "All processes completed successfully!"
    echo "Check the output/ directory for logs and results."
    echo "Output files copied to: $SLURM_SUBMIT_DIR/output/"
else
    echo ""
    echo "Process failed with exit code: $FINAL_EXIT_CODE"
    echo "Check the log files for error details."
    echo "Partial output files copied to: $SLURM_SUBMIT_DIR/output/"
fi

exit $FINAL_EXIT_CODE