#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=results/profile_job_%j.out
#SBATCH --error=results/profile_job_%j.err
#SBATCH --job-name=cs380_project_profile

# 1. Load Environment
module purge
module load nvidia-sdk
module load cuda/11.8  # Fallback often needed despite nvidia-sdk

# 2. Build (Ensure we have symbols for profiling)
# We add -lineinfo to nvcc flags for better profiling data
echo "Building with line info..."
cd build
# Clean previous build to force recompilation with new flags
rm -f CMakeCache.txt
cmake .. -DCMAKE_CUDA_FLAGS="-lineinfo -O3 -use_fast_math"
make -j

# Location of executable
EXE="./bin/mixed_precision_ir"

echo "Starting Profiling Job on $(hostname)"
echo "------------------------------------------------"

# 3. Nsight Systems (Timeline & Concurrency)
# Good for: Seeing CPU/GPU overlap, memory transfer times, and overall timeline.
echo "Running Nsight Systems (Timeline)..."
nsys profile \
    --trace=cuda,cublas,cudnn,nvtx,osrt \
    --output=report_timeline_${SLURM_JOB_ID} \
    --force-overwrite=true \
    $EXE

echo "------------------------------------------------"

# 4. Nsight Compute (Kernel Metrics)
# Good for: Roofline model, Memory Throughput, Tensor Core Utilization.
# Warning: This is slow! We target N=8192 to get good data without waiting forever.
# We limit the number of kernels profiled to avoid 100GB files.
echo "Running Nsight Compute (Kernel Analysis)..."
ncu \
    --set full \
    --target-processes all \
    -o report_kernels_${SLURM_JOB_ID} \
    --force-overwrite \
    --kernel-name-base=function \
    -k regex:gemm \
    --launch-count 100 \
    $EXE --profile

echo "------------------------------------------------"
echo "Profiling Complete."
echo "Download the .nsys-rep and .ncu-rep files to your local machine to view."
