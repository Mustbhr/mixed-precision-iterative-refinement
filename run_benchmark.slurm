#!/bin/bash
#SBATCH --job-name=precision_bench
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --partition=batch
#SBATCH --output=results/benchmark_%j.txt
#SBATCH --error=results/benchmark_err_%j.txt

module purge
module load cuda
module load cmake

cd $SLURM_SUBMIT_DIR

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4 precision_benchmark

if [ $? -eq 0 ]; then
    echo "Running Precision Benchmark..."
    ./bin/precision_benchmark
else
    echo "Build failed!"
    exit 1
fi
