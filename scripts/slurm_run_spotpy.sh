#!/bin/bash
#SBATCH --account=def-nicoleli
#SBATCH --time=${RUNTIME:-01:00:00}
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=64000M
#SBATCH --mail-user=${EMAIL}
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Create directory for storing slurm_logs
mkdir -p slurm_logs

module load StdEnv/2020 gcc/9.3.0 cuda/11.0 python/3.11 mpi4py/4.0.0 || { echo "Module load failed"; exit 1; }

# PACKAGE DIRECTORY AT FAILURE OR SUCCESS
current_date=$(date +"%Y-%m-%d")
tarball_name="param_opt_${current_date}_${SLURM_JOB_ID}_rope.tar.gz"

# Create directory for storing tarballs if it doesn't exist
mkdir -p finished_runs
echo "The package directory will be saved to the finished_runs directory"


# trap to package directory on any exit (success or failure) excluding env
function package_dir()
{
  echo "Packaging directory (trap)..."
  cd "$SLURM_TMPDIR"
  # Copy the slurm log to the tarball
  cp "$SLURM_SUBMIT_DIR/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" .
  cp "$SLURM_SUBMIT_DIR/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" .
  tar --exclude="env" --exclude="./env" --exclude="$SLURM_SUBMIT_DIR/finished_runs" -czf "$SLURM_SUBMIT_DIR/finished_runs/$tarball_name" .
  echo "Packaged directory into: $SLURM_SUBMIT_DIR/finished_runs/$tarball_name"
  exit
}

trap 'package_dir' EXIT


echo "SLURM_TMPDIR: $SLURM_TMPDIR"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# Copy required files to SLURM_TMPDIR
echo "Copying files to SLURM_TMPDIR..."
rsync -av --exclude='slurm_logs' --exclude='finished_runs' "$SLURM_SUBMIT_DIR/" "$SLURM_TMPDIR/" || { echo "Failed to copy files"; exit 1; }
cd "$SLURM_TMPDIR" || { echo "Failed to change to SLURM_TMPDIR"; exit 1; }

# Create output directory if it doesn't exist
mkdir -p output
mkdir -p "$SLURM_SUBMIT_DIR/output"

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
pip install --no-index pandas click numpy pathos spotpy scikit-learn mpi4py || { echo "Failed to install Python dependencies"; exit 1; }

pip freeze > installed_requirements.txt

export OMP_NUM_THREADS=32
export OMP_NESTED=TRUE

# Set script directory and project root (now relative to SLURM_TMPDIR)
SCRIPT_DIR="$SLURM_TMPDIR"
PROJECT_ROOT="$SLURM_TMPDIR"

echo "Project root: $PROJECT_ROOT"
echo "Running ROPE optimization with simulation..."

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
NUM_ITERATIONS=800
PARALLEL="mpc"
SAVE_RUNS=false

# Algorithm selection
ALGORITHM="rope"  # Default algorithm

# ROPE-specific parameters
SUBSETS=6
NUM_REPS_FIRST_RUN=400
PERCENTAGE_FIRST_RUN=0.1
PERCENTAGE_FOLLOWING_RUNS=0.1

# Monte Carlo-specific parameters
MC_REPETITIONS=1000

# SCE-UA-specific parameters
SCEUA_NGS=20
SCEUA_KSTOP=100
SCEUA_PEPS=0.0000001
SCEUA_PCENTO=0.0000001

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --algorithm|-a)
            ALGORITHM="$2"
            shift 2
            ;;
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
        --save-runs|-sr)
            SAVE_RUNS="$2"
            shift 2
            ;;
        # ROPE-specific parameters
        --repetitions-first-run|-rf)
            NUM_REPS_FIRST_RUN="$2"
            shift 2
            ;;
        --subsets|-sbs)
            SUBSETS="$2"
            shift 2
            ;;
        --percentage-first-run|-pfr)
            PERCENTAGE_FIRST_RUN="$2"
            shift 2
            ;;
        --percentage-following-runs|-pfrs)
            PERCENTAGE_FOLLOWING_RUNS="$2"
            shift 2
            ;;
        # Monte Carlo-specific parameters
        --mc-repetitions|-mcr)
            MC_REPETITIONS="$2"
            shift 2
            ;;
        # SCE-UA-specific parameters
        --sceua-ngs|-ngs)
            SCEUA_NGS="$2"
            shift 2
            ;;
        --sceua-kstop|-kstop)
            SCEUA_KSTOP="$2"
            shift 2
            ;;
        --sceua-peps|-peps)
            SCEUA_PEPS="$2"
            shift 2
            ;;
        --sceua-pcento|-pcento)
            SCEUA_PCENTO="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "General Options:"
            echo "  -a, --algorithm       Algorithm to use (rope|mc|sceua)"
            echo "  -l, --log-level       Set logging level (DEBUG|INFO|WARNING|ERROR|CRITICAL)"
            echo "  -pr, --param-ranking  Parameter ranking method (random_forest|morris)"
            echo "  -pn, --param-num      Number of parameters to rank (default: 5)"
            echo "  -i, --num-iterations  Number of optimization iterations (default: 800)"
            echo "  -p, --parallel        Parallelization method (mpc|mpi|seq)"
            echo "  -sr, --save-runs      Save individual simulation runs (default: false)"
            echo ""
            echo "ROPE-specific Options:"
            echo "  -rf, --repetitions-first-run     Number of repetitions for the first run (default: 400)"
            echo "  -sbs, --subsets                  Number of subsets for the ROPE sampler (default: 6)"
            echo "  -pfr, --percentage-first-run     Percentage of the first run (default: 0.1)"
            echo "  -pfrs, --percentage-following-runs  Percentage of following runs (default: 0.1)"
            echo ""
            echo "Monte Carlo-specific Options:"
            echo "  -mcr, --mc-repetitions           Number of Monte Carlo repetitions (default: 1000)"
            echo ""
            echo "SCE-UA-specific Options:"
            echo "  -ngs, --sceua-ngs                Number of complexes (default: 7)"
            echo "  -kstop, --sceua-kstop            Number of shuffling loops (default: 3)"
            echo "  -peps, --sceua-peps              Convergence tolerance (default: 0.1)"
            echo "  -pcento, --sceua-pcento          Percentage for convergence (default: 0.1)"
            echo ""
            echo "  -h, --help                       Show this help message"
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
echo "  Algorithm: $ALGORITHM"
echo "  Log Level: $LOG_LEVEL"
echo "  Parameter Ranking: $PARAM_RANKING"
echo "  Number of Parameters: $PARAM_NUM"
echo "  Iterations: $NUM_ITERATIONS"
echo "  Parallelization: $PARALLEL"
echo "  Save Runs: $SAVE_RUNS"

case $ALGORITHM in
    rope)
        echo "  ROPE Parameters:"
        echo "    Repetitions First Run: $NUM_REPS_FIRST_RUN"
        echo "    Subsets: $SUBSETS"
        echo "    Percentage First Run: $PERCENTAGE_FIRST_RUN"
        echo "    Percentage Following Runs: $PERCENTAGE_FOLLOWING_RUNS"
        ;;
    mc)
        echo "  Monte Carlo Parameters:"
        echo "    MC Repetitions: $MC_REPETITIONS"
        ;;
    sceua)
        echo "  SCE-UA Parameters:"
        echo "    NGS (complexes): $SCEUA_NGS"
        echo "    KSTOP: $SCEUA_KSTOP"
        echo "    PEPS: $SCEUA_PEPS"
        echo "    PCENTO: $SCEUA_PCENTO"
        ;;
    *)
        echo "Error: Unknown algorithm '$ALGORITHM'. Use: rope, mc, or sceua"
        exit 1
        ;;
esac

cd "$PROJECT_ROOT"

echo "Starting $ALGORITHM optimization..."
echo "About to run $ALGORITHM from directory: $(pwd)"

# Build common arguments
COMMON_ARGS=(
    --log-level "$LOG_LEVEL"
    --param-ranking "$PARAM_RANKING"
    --param-num "$PARAM_NUM"
    --num-iterations "$NUM_ITERATIONS"
    --run-dir-parent "$RUN_DIR"
    --sensitivity-analysis-csv "$SENSITIVITY_CSV"
    --experimental-data-csv "$EXPERIMENTAL_CSV"
    --config-file-dir "$CONFIG_DIR"
    --bin-dir "$BIN_DIR"
    --parallel "$PARALLEL"
    --save-runs "$SAVE_RUNS"
)

# Run the appropriate algorithm
case $ALGORITHM in
    rope)
        python -m src.calibration.run_rope \
            "${COMMON_ARGS[@]}" \
            --repetitions-first-run "$NUM_REPS_FIRST_RUN" \
            --subsets "$SUBSETS" \
            --percentage-first-run "$PERCENTAGE_FIRST_RUN" \
            --percentage-following-runs "$PERCENTAGE_FOLLOWING_RUNS" \
            > output/output.txt 2>&1
        ;;
    mc)
        echo "Not implemented yet."
        ;;
    sceua)
        echo "Not implemented yet."
        ;;
esac

OPTIMIZATION_EXIT_CODE=$?
echo "$ALGORITHM completed with exit code: $OPTIMIZATION_EXIT_CODE"
echo "Contents after $ALGORITHM:"
ls -la output/ 2>/dev/null || echo "No output directory after $ALGORITHM"

if [ $OPTIMIZATION_EXIT_CODE -eq 0 ]; then
    echo "$ALGORITHM optimization completed successfully, running parameter extraction..."
    python -m src.post_calibration.parameter_extraction >> output/output.txt 2>&1
    
    echo "Running validation..."
    python -m src.post_calibration.validation \
        --param-file "$PROJECT_ROOT/output/${ALGORITHM}_abm_optimization.csv" \
        --run-dir "$RUN_DIR" \
        --bin-dir "$BIN_DIR" \
        --config-dir "$CONFIG_DIR" \
        --sensitivity-analysis-csv "$SENSITIVITY_CSV" \
        --param-ranking "$PARAM_RANKING" \
        --param-num "$PARAM_NUM" \
        --exp-data "$EXPERIMENTAL_CSV" \
        --use-csv >> output/output.txt 2>&1

    FINAL_EXIT_CODE=$?
else
    echo "$ALGORITHM optimization failed with exit code: $OPTIMIZATION_EXIT_CODE"
    FINAL_EXIT_CODE=$OPTIMIZATION_EXIT_CODE
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