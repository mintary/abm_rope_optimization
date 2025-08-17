# ROPE ABM optimization

## SLURM submission

The scripts for submission to the SLURM workload manager are found in the `scripts/` directory. It's very important that you run the script such that your current working directory is the project root! In the future, we may make it possible to run it from everywhere.

### ROPE Algorithm

To submit a ROPE optimization job:

```bash
export EMAIL=emily.wang10@mail.mcgill.ca
sbatch --mail-user=$EMAIL scripts/slurm_run_spotpy.sh --algorithm rope
```

Adjusting ROPE settings:

```bash
export EMAIL=emily.wang10@mail.mcgill.ca
sbatch \
    --mail-user=$EMAIL \
    --time=05:00:00 \
    scripts/slurm_run_spotpy.sh \
    --algorithm rope \
    --num-iterations 1000 \
    --repetitions-first-run 500 \
    --subsets 8 \
    --percentage-first-run 0.25 \
    --percentage-following-runs 0.15
```

## Passing the Number of Parameters to Optimize

You can specify the number of parameters to optimize in the ROPE algorithm using the `--param-num` (or `-pn`) option. This is supported both in SLURM scripts and when running locally. For example:

**SLURM submission:**

```bash
export EMAIL=emily.wang10@mail.mcgill.ca
sbatch --mail-user=$EMAIL scripts/slurm_run_spotpy.sh --algorithm rope --param-num 4
```

**Local run:**

```bash
python src/calibration/run_rope.py --param-num 4
```

This will optimize 4 parameters, as selected by the parameter ranking method (default: random_forest). You can also use the `--param-ranking` option to change the ranking method (e.g., `--param-ranking morris`).

### Mock simulation with SLURM

You can also run a shorter script (3 minute runtime) that steps through the workflow with a test simulation.

To submit a job:

```bash
mkdir -p slurm_logs
export EMAIL=emily.wang10@mail.mcgill.ca
sbatch --mail-user=$EMAIL --output ./slurm_logs/slurm-%j.out scripts/slurm_run_rope_mock.sh
```

### To-dos:

#### Optimization

- [ ] Record runtimes and memory taken for each number of iterations (100, 200, 300...)
- [x] Have another SPOTPY algorithm (i.e. pure Monte Carlo sampling) for comparison
- [ ] Automate optimizing for every biomarker separately, then running ROPE for all of them
- [ ] Review the validation process to ensure it matches the manuscript

#### QOL

- [ ] Better documentation of the command-line arguments (automated documentation?)
- [ ] Ability to submit batch jobs
- [ ] Automated flow for visualizing results and creating a report for that run

## Environment

### Windows

Run the following commands:

```bash
python -m venv venv
venv\Scripts\Activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Mac/Linux

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Mock simulation

The mock simulation (`mock_testRun`) is a C++ program designed to test and validate the ROPE optimization algorithm before applying it to the actual ABM simulation. It simulates the behavior of the real ABM by generating predictable biomarker data based on input parameters.

We use this to test the optimization process without running the computationally expensive ABM world simulation. All parameters should converge towards `1`.

### Architecture

### Target Values

The simulation targets experimental data from three scaffold configurations:

**Cell Counts (Total Fibroblasts)**:

- GH10: [90, 90] cells at [72h, 144h]
- GH2: [130, 87] cells at [72h, 144h]
- GH5: [65, 85] cells at [72h, 144h]

**Collagen Levels**:

- GH10: [647000, 428000] at [72h, 144h]
- GH2: [551000, 1000000] at [72h, 144h]
- GH5: [912000, 649000] at [72h, 144h]

#### Build Process

```bash
# Navigate to mock simulation directory
cd mock_testRun

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the executable
cmake --build .

# The executable will be copied to mock_testRun/bin/
```

#### Running the Mock Simulation

**Basic Usage**:

```bash
# Run with default parameters (from Sample.txt)
./bin/testRun

# Specify configuration file
./bin/testRun --inputfile config_GH10.txt

# Set custom parameters
./bin/testRun --numticks 289 --inputfile config_GH2.txt
```

**Command Line Arguments**:

- `--numticks <int>`: Number of simulation time steps (default: 289)
- `--inputfile <string>`: Configuration file name (affects target selection)
- `--wxw <double>`: X-dimension weight (default: 0.6)
- `--wyw <double>`: Y-dimension weight (default: 0.6)
- `--wzw <double>`: Z-dimension weight (default: 0.6)

### Parameter File Format

The `Sample.txt` file should contain tab-separated parameter values:

```
0.8 1.2 0.9 1.1 1.0
```

This represents: `[p0, p1, p2, p3, p4]`

## Running ROPE algorithm with mock simulation locally

The `run_rope_mock.sh` script provides a convenient way to run the ROPE optimization algorithm with the mock simulation. This script automates the process of building the mock simulation (if needed), setting up the necessary directories, and running the optimization with configurable parameters.

### Basic Usage

```bash
./scripts/run_rope_mock.sh
```

### Command Line Options

The script supports the following options:

| Option                 | Description                            | Default Value   |
| ---------------------- | -------------------------------------- | --------------- |
| `-l, --log-level`      | Set logging level                      | `DEBUG`         |
| `-pr, --param-ranking` | Parameter ranking method               | `random_forest` |
| `-pn, --param-num`     | Number of parameters to rank           | `5`             |
| `-i, --num-iterations` | Number of optimization iterations      | `800`           |
| `-p, --parallel`       | Parallelization method (mpc, mpi, seq) | `mpc`           |
| `-h, --help`           | Show help information                  |                 |

### Examples

Run with default settings:

```bash
./scripts/run_rope_mock.sh
```

Run with custom settings:

```bash
./scripts/run_rope_mock.sh --log-level INFO --param-num 4 --num-iterations 500 --parallel seq
```

Use the Morris method for parameter ranking:

```bash
./scripts/run_rope_mock.sh --param-ranking morris
```

### Output Files

After the script completes, it will generate several output files:

- `optimization_main.log`: Main process log messages
- `optimization_progress.log`: Detailed optimization progress tracking
- `optimization_report.json`: Final optimization report in JSON format
- `rope_abm_optimization.csv`: SPOTPY results database

It will also run validation, generating these files in `validation/`:

- `validation_metrics.json`: Metrics (including C.I. for each biomarker)
- `validation_results.json`: Actual values of outputed biomarkers from the simulation with those parameters
- `validation.log`: Validation process log messages
