#!/bin/bash

# Set script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root: $PROJECT_ROOT"
echo "Running ROPE optimization with mock simulation..."

# Ensure mock simulation is built
MOCK_DIR="$PROJECT_ROOT/mock_testRun"
if [ ! -f "$MOCK_DIR/bin/testRun" ] && [ ! -f "$MOCK_DIR/bin/testRun.exe" ]; then
    echo "Building mock simulation..."
    cd "$MOCK_DIR"
    
    # Create build directory if it doesn't exist
    if [ ! -d "build" ]; then
        mkdir build
    fi
    
    cd build
    cmake ..
    cmake --build .
    
    # Check if build was successful
    if [ ! -f "$MOCK_DIR/bin/testRun" ] && [ ! -f "$MOCK_DIR/bin/testRun.exe" ]; then
        echo "Error: Failed to build mock simulation"
        exit 1
    fi
    
    echo "Mock simulation built successfully"
    cd "$PROJECT_ROOT"
else
    echo "Mock simulation already built"
fi

# Create output directory for simulation runs
RUN_DIR="$PROJECT_ROOT/output/rope_runs"
mkdir -p "$RUN_DIR"

# Set up paths for the CLI arguments
SENSITIVITY_CSV="$PROJECT_ROOT/input/sensitivity_analysis_mock.csv"
EXPERIMENTAL_CSV="$PROJECT_ROOT/input/experimental.csv"
CONFIG_DIR="$PROJECT_ROOT/input/configFiles"
BIN_DIR="$PROJECT_ROOT/mock_testRun/bin"

# Validate required files exist
if [ ! -f "$SENSITIVITY_CSV" ]; then
    echo "Error: Sensitivity analysis CSV not found at $SENSITIVITY_CSV"
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
            echo "  -i, --num-iterations  Number of optimization iterations (default: 100)"
            echo "  -p, --parallel        Parallelization method (mpc|mpi|seq)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "This script runs ROPE optimization with the mock simulation for testing."
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
    --parallel "$PARALLEL"



EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "ROPE optimization completed successfully!"
    echo "Check the following files in output/ for results:"
    echo "  - optimization_main.log (main log)"
    echo "  - optimization_progress.log (progress tracking)"
    echo "  - optimization_report.json (final report)"
    echo "  - rope_abm_optimization.csv (SPOTPY results database with best parameter sets, including errors)"
else
    echo ""
    echo "ROPE optimization failed with exit code: $EXIT_CODE"
    echo "Check the log files for error details."
fi

exit $EXIT_CODE
