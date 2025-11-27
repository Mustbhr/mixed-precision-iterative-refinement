# Phase 1: Building and Running on IBEX

## Quick Start on IBEX

### 1. SSH to IBEX
```bash
ssh YOUR_KAUST_ID@ilogin.ibex.kaust.edu.sa
```

### 2. Clone/Pull Your Repository
```bash
# If first time
git clone https://github.com/YOUR_USERNAME/mixed-precision-iterative-refinement.git
cd mixed-precision-iterative-refinement

# If already cloned
cd mixed-precision-iterative-refinement
git pull
```

### 3. Option A: Interactive Build and Test (Recommended for First Time)

```bash
# Request interactive GPU session
srun --time=00:30:00 --gres=gpu:1 --mem=16G --pty bash

# Load modules
module load cuda/11.8 cmake/3.20

# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# Run
./bin/mixed_precision_ir

# Exit when done
exit
```

### 4. Option B: Batch Job Submission

```bash
# Make sure you're in the project root
cd ~/mixed-precision-iterative-refinement

# Submit the job
sbatch run_phase1.slurm

# Check job status
squeue -u $USER

# View output (once job starts/completes)
tail -f results/phase1_output_JOBID.txt

# Or view after completion
cat results/phase1_output_*.txt | tail -100
```

## Expected Output

You should see something like:

```
========================================
Mixed-Precision Iterative Refinement
Phase 1: FP64 Baseline Solver
CS380 GPU Programming Project
========================================

Found 1 CUDA device(s)

Device 0: Tesla V100-SXM2-32GB
  Compute Capability: 7.0
  Global Memory: 32510 MB
  ...
  ✓ Tensor Cores: Supported (Volta - FP16)

========================================
Test 1: Identity Matrix (4×4)
========================================

Matrix A (Identity):
...
✓ Solver succeeded
Time: 0.234 ms
Relative error: 2.31e-15
✓ PASS: Accuracy excellent!

========================================
Test 2: Random Matrix (64×64)
========================================
...
✓ PASS: Accuracy excellent!

========================================
Benchmark: FP64 Performance Scaling
========================================

    Size    Time(ms)     GFLOPS    Rel. Error
-----------------------------------------------
      64       0.156      11.234    2.45e-14
     128       0.421      19.823    1.87e-14
     256       2.134      32.156    3.12e-14
     512      14.567      36.891    2.94e-14
    1024      87.234      82.145    3.45e-14

✓ Phase 1 Complete!
```

## What to Check

### ✅ Build Success
- CMake finds CUDA
- All libraries link correctly
- No compilation errors

### ✅ GPU Detection
- Device info shows your GPU
- Compute capability displayed
- Tensor Core support indicated

### ✅ Correctness Tests Pass
- Identity matrix: error < 1e-14
- Random matrices: error < 1e-10
- Poisson matrix: error < 1e-10

### ✅ Performance Reasonable
- Time scales roughly as O(n³)
- GFLOPS increases with matrix size
- No crashes on larger sizes

## Troubleshooting

### Build Errors

**Error: "CUDA not found"**
```bash
module load cuda/11.8
# Check it's loaded
module list
```

**Error: "cuSOLVER not found"**
```bash
# Should be in CUDA toolkit, check version
nvcc --version
# Try cuda/12.x if 11.x doesn't work
```

**Error: "Compute capability mismatch"**
```bash
# Edit CMakeLists.txt if needed
# Add your GPU's compute capability
# V100: 70, A100: 80
```

### Runtime Errors

**Error: "No CUDA device"**
```bash
# Make sure you requested GPU
srun --gres=gpu:1 --pty bash
nvidia-smi  # Should show GPU
```

**Error: "Out of memory"**
```bash
# Request more memory or reduce test sizes
srun --gres=gpu:1 --mem=32G --pty bash
```

**Error: "Solver failed"**
```bash
# Check matrix conditioning
# Try smaller sizes first
# Check GPU memory usage
```

## Next Steps

Once Phase 1 works:

1. **Analyze the results:**
   - What's your FP64 baseline performance?
   - What accuracy do you achieve?
   - How does performance scale?

2. **Document your findings:**
   - Save the output to `results/phase1_baseline.txt`
   - Note your GPU model and performance
   - This is your reference for Phase 2

3. **Commit your results:**
```bash
# On IBEX
cp results/phase1_output_*.txt results/phase1_baseline.txt

# Then on your Mac after pulling
git add results/phase1_baseline.txt
git commit -m "Add Phase 1 baseline results"
git push
```

4. **Ready for Phase 2:**
   - You have working baseline
   - You understand the accuracy target
   - You know the performance to beat
   - Time to implement mixed-precision iterative refinement!

## Performance Analysis

Record these numbers for your report:

| Matrix Size | Time (ms) | GFLOPS | Relative Error |
|-------------|-----------|--------|----------------|
| 64          |           |        |                |
| 128         |           |        |                |
| 256         |           |        |                |
| 512         |           |        |                |
| 1024        |           |        |                |

**Questions to answer:**
1. How does time scale with n? (Should be O(n³))
2. What's the achieved GFLOPS? (Compare to theoretical peak)
3. Is accuracy consistent across sizes?
4. Does Poisson matrix behave differently than random?

Good luck! 🚀

