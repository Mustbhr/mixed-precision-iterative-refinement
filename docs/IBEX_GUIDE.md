# IBEX Setup and Usage Guide

This guide covers how to work with KAUST's IBEX HPC cluster for GPU computing.

## Initial Setup

### 1. SSH Connection
```bash
ssh YOUR_KAUST_ID@ilogin.ibex.kaust.edu.sa
```

### 2. Clone Your Repository
```bash
cd $HOME
git clone https://github.com/YOUR_USERNAME/mixed-precision-iterative-refinement.git
cd mixed-precision-iterative-refinement
```

### 3. Check Available Modules
```bash
module avail cuda    # See available CUDA versions
module avail cmake   # See available CMake versions
module avail gcc     # See available GCC versions
```

### 4. Load Required Modules
```bash
# Add to your ~/.bashrc for persistence
module load cuda/11.8      # or latest version
module load cmake/3.20     # or latest
module load gcc/10.2.0     # if needed
```

## GPU Information on IBEX

### Available GPU Types (as of 2024)
- **V100 (Volta):** 32GB memory, Compute Capability 7.0
  - Tensor Core support: FP16
  - Good for general GPU computing
  
- **A100 (Ampere):** 40GB/80GB memory, Compute Capability 8.0
  - Tensor Core support: FP16, TF32, BF16
  - Best for this project (TF32 support!)
  
- **RTX 6000 (Turing):** 24GB memory, Compute Capability 7.5
  - Tensor Core support: FP16, INT8

### Check GPU Availability
```bash
# See which GPUs are available
sinfo -p gpu -o "%20N %10c %10m %25f %10G"

# Check current GPU queue
squeue -p gpu
```

## Building Your Code

### Interactive Build Session
```bash
# Request interactive session with GPU
srun --time=01:00:00 --gres=gpu:1 --mem=16G --pty bash

# Load modules (if not in .bashrc)
module load cuda/11.8 cmake/3.20

# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# Test
./bin/mixed_precision_ir

# Exit interactive session when done
exit
```

### Building from Login Node (No GPU Needed)
```bash
# You can compile on login node, just don't run GPU code
module load cuda/11.8 cmake/3.20
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

## Running Jobs

### Option 1: Interactive Session (for debugging)
```bash
# Request GPU node
srun --time=02:00:00 \
     --gres=gpu:v100:1 \
     --mem=32G \
     --pty bash

# Run your program
cd /path/to/your/project/build
./bin/mixed_precision_ir

# Can run multiple tests, debug, etc.
```

### Option 2: Batch Job (for benchmarking)

Create `job_v100.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=mpir_v100
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --partition=batch
#SBATCH --output=results/output_v100_%j.txt
#SBATCH --error=results/error_v100_%j.txt

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load modules
module purge
module load cuda/11.8
module load cmake/3.20

# Print GPU info
nvidia-smi

# Go to project directory
cd $SLURM_SUBMIT_DIR

# Run benchmark
./build/bin/mixed_precision_ir

echo "End time: $(date)"
```

Submit job:
```bash
sbatch job_v100.slurm
```

Create `job_a100.slurm` (for A100, replace gres line):
```bash
#SBATCH --gres=gpu:a100:1
```

### Monitor Jobs
```bash
# Check your jobs
squeue -u $USER

# Check job details
scontrol show job JOB_ID

# Cancel a job
scancel JOB_ID

# View output (while job is running)
tail -f results/output_v100_JOBID.txt
```

## Nsight Compute Profiling

### Interactive Profiling
```bash
# Request GPU session
srun --time=01:00:00 --gres=gpu:v100:1 --mem=16G --pty bash

# Load modules
module load cuda/11.8

# Profile your application
ncu --set full -o profile_report ./bin/mixed_precision_ir

# View report (if X11 forwarding is enabled)
ncu-ui profile_report.ncu-rep

# Or export to CSV for analysis
ncu --csv --log-file profile.csv ./bin/mixed_precision_ir
```

### Batch Profiling
Add to your SLURM script:
```bash
# Profile specific kernels
ncu --kernel-name "regex:gemm" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    -o profile_output \
    ./bin/mixed_precision_ir
```

## Best Practices

### 1. Don't Run on Login Nodes
❌ **Never do this:**
```bash
# On login node
./my_gpu_program  # Will fail or get you in trouble!
```

✅ **Always do this:**
```bash
# Request interactive session first
srun --gres=gpu:1 --pty bash
./my_gpu_program
```

### 2. Start with Short Time Limits
- Request 1 hour first to test
- Then increase for longer benchmarks
- IBEX may prioritize shorter jobs

### 3. Clean Up Output Files
```bash
# Results can accumulate quickly
ls -lh results/
rm results/old_output_*.txt
```

### 4. Use Job Arrays for Parameter Sweeps
```bash
#!/bin/bash
#SBATCH --array=1-10
#SBATCH --job-name=mpir_sweep

# Use $SLURM_ARRAY_TASK_ID to vary matrix size
MATRIX_SIZE=$((512 * $SLURM_ARRAY_TASK_ID))
./bin/mixed_precision_ir --size $MATRIX_SIZE
```

## Useful IBEX Commands

```bash
# Your quota
quota -s

# Your disk usage
du -sh $HOME/*

# Available partitions
sinfo

# Detailed node info
scontrol show node NODE_NAME

# Job efficiency after completion
seff JOB_ID

# Your job history
sacct -u $USER --format=JobID,JobName,State,Time,Elapsed
```

## File Transfer

### From Mac to IBEX
```bash
# Single file
scp file.txt YOUR_ID@ilogin.ibex.kaust.edu.sa:~/project/

# Directory
scp -r results/ YOUR_ID@ilogin.ibex.kaust.edu.sa:~/project/

# Or use rsync
rsync -avz --progress results/ YOUR_ID@ilogin.ibex.kaust.edu.sa:~/project/results/
```

### From IBEX to Mac
```bash
# Download results
scp YOUR_ID@ilogin.ibex.kaust.edu.sa:~/project/results/*.csv ./results/

# Or use rsync
rsync -avz --progress YOUR_ID@ilogin.ibex.kaust.edu.sa:~/project/results/ ./results/
```

## Recommended Workflow

1. **Develop on Mac:**
   - Edit code in Cursor/VS Code
   - Commit and push to GitHub

2. **Build and Test on IBEX:**
   ```bash
   ssh ibex
   cd ~/project
   git pull
   cd build && make -j8
   srun --gres=gpu:1 --time=00:30:00 --pty bash
   ./bin/mixed_precision_ir  # Quick test
   exit
   ```

3. **Run Benchmarks:**
   ```bash
   sbatch job_v100.slurm
   squeue -u $USER  # Monitor
   ```

4. **Download Results:**
   ```bash
   # On your Mac
   scp ibex:~/project/results/*.csv ./results/
   ```

5. **Analyze Locally:**
   - Plot results on your Mac
   - Update documentation
   - Commit and push

## Troubleshooting

### "Out of Memory" Error
- Reduce matrix size
- Request more memory: `--mem=64G`
- Check for memory leaks (cudaFree all allocations)

### "CUDA Error: Invalid Device Function"
- GPU compute capability mismatch
- Update CMakeLists.txt with correct architectures
- For V100 (7.0): `set(CMAKE_CUDA_ARCHITECTURES 70)`
- For A100 (8.0): `set(CMAKE_CUDA_ARCHITECTURES 80)`

### Job Stuck in Queue
- Check: `squeue -u $USER`
- Try shorter time limit
- Try different GPU type
- Check if you have running jobs (limits apply)

### Module Not Found
```bash
module spider cuda  # Search for module
module spider cuda/11.8  # How to load specific version
```

## Contact

For IBEX issues:
- Email: ibex@kaust.edu.sa
- Documentation: https://www.hpc.kaust.edu.sa/ibex

For project questions:
- Instructor: Prof. Markus Hadwiger
- TA: Peter Rautek

