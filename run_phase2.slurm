#!/bin/bash
#SBATCH --job-name=phase2_ir
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --output=results/phase2_output_%j.txt
#SBATCH --error=results/phase2_error_%j.txt

# Print job information
echo "==========================================="
echo "Phase 2: Mixed-Precision Iterative Refinement"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load required modules
module purge
module load cuda/11.8
module load cmake/3.20

# Print module info
echo "Loaded modules:"
module list
echo ""

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Build the project
echo "Building project..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    
    # Run the program
    echo "Running Phase 2 tests..."
    echo "==========================================="
    ./bin/mixed_precision_ir
    
    echo ""
    echo "==========================================="
    echo "Job completed!"
else
    echo "Build failed!"
    exit 1
fi

echo "End time: $(date)"
