# Getting Started Guide

## Step 1: Create GitHub Repository

1. **Go to GitHub** and create a new repository:
   - Repository name: `mixed-precision-iterative-refinement` (or your choice)
   - Description: "Mixed-Precision Iterative Refinement Solver using GPU Tensor Cores - CS380 Project"
   - Keep it **Public** (unless you prefer private)
   - **Don't** initialize with README (we already have one)

2. **Link your local repository** to GitHub:
   ```bash
   cd /Users/mustafa/Desktop/KAUST/CS380/CS380-Project
   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

3. **Add a nice description** on GitHub with topics:
   - cuda
   - gpu-computing
   - tensor-cores
   - numerical-linear-algebra
   - mixed-precision
   - iterative-refinement

## Step 2: Learning Path

### A. Understand the Theory (1-2 days)

Read the `docs/THEORY.md` file I created. Key concepts to understand:

1. **Linear system solving:** LU factorization, forward/backward substitution
2. **Iterative refinement:** Why computing residuals in higher precision helps
3. **Precision formats:** FP16, TF32, FP32, FP64 trade-offs
4. **Tensor Cores:** What they are and why they're fast

**Recommended readings:**
- [ ] Haidar et al. SC'18 paper (main inspiration for this project)
- [ ] NVIDIA Tensor Core documentation
- [ ] Review LU factorization if needed (any numerical analysis textbook)

### B. CUDA Refresher (1 day if needed)

Make sure you're comfortable with:
- [ ] CUDA kernel basics (threads, blocks, grids)
- [ ] Shared memory
- [ ] cuBLAS library usage
- [ ] Memory transfers (host ↔ device)

**Resources:**
- NVIDIA CUDA Programming Guide
- CUDA by Example (book)

### C. Study Tensor Core Programming (2-3 days)

Choose your API (I recommend starting with WMMA, then moving to CUTLASS):

**WMMA (Simpler, good for learning):**
- [ ] Read: https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
- [ ] Study the fragment types and synchronization
- [ ] Understand 16×16×16 tile sizes

**CUTLASS (More powerful):**
- [ ] Clone CUTLASS: `git clone https://github.com/NVIDIA/cutlass.git`
- [ ] Study examples in `examples/` directory
- [ ] Read quickstart guide
- [ ] Understand the template-based design

### D. cuBLAS and cuSOLVER (1 day)

You'll use these for baseline and comparison:
- [ ] cuBLAS: Matrix operations (GEMM, GEMV)
- [ ] cuSOLVER: LU factorization (getrf), triangular solves (getrs)

## Step 3: Development Workflow

Since you're developing on Mac and running on IBEX:

### On MacOS (Your Laptop):
```bash
# 1. Edit code in Cursor/VS Code
# 2. Commit changes
git add .
git commit -m "Description of changes"
git push

# 3. Can't compile CUDA on Mac, but you can:
# - Write code with syntax highlighting
# - Plan and document
# - Work on CPU portions
```

### On IBEX (KAUST HPC):
```bash
# 1. SSH into IBEX
ssh YOUR_USERNAME@ilogin.ibex.kaust.edu.sa

# 2. Clone/pull your repository
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME
# or if already cloned:
git pull

# 3. Load modules
module load cuda/11.8  # or latest version
module load cmake/3.20  # or latest

# 4. Build
mkdir -p build && cd build
cmake ..
make -j8

# 5. Submit jobs (don't run on login node!)
sbatch job_script.slurm
```

## Step 4: Implementation Roadmap

### Phase 1: Hello CUDA & Setup (Week 1)

**Goal:** Get comfortable with the environment

Tasks:
- [ ] Create a simple "Hello from GPU" program
- [ ] Test compilation on IBEX
- [ ] Set up cuBLAS linkage
- [ ] Implement matrix multiplication with cuBLAS
- [ ] Create matrix generation utilities (random, identity, Poisson)

**Deliverable:** Basic working CUDA program that can multiply matrices

### Phase 2: Pure FP64 Baseline (Week 1-2)

**Goal:** Implement a working solver in pure FP64 using cuBLAS/cuSOLVER

Tasks:
- [ ] Implement LU factorization using `cusolverDnDgetrf`
- [ ] Implement triangular solve using `cusolverDnDgetrs`
- [ ] Implement residual computation: r = b - Ax
- [ ] Implement solution norm and residual norm
- [ ] Test on small matrices (verify correctness)

**Deliverable:** Working FP64 solver

### Phase 3: Mixed-Precision Iterative Refinement (Week 2-3)

**Goal:** Implement the core algorithm

Tasks:
- [ ] Implement FP32 factorization, FP64 residual (simpler version)
- [ ] Implement iterative refinement loop
- [ ] Add convergence checking
- [ ] Test and verify convergence
- [ ] Compare accuracy vs. pure FP64

**Deliverable:** Working mixed-precision solver achieving FP64 accuracy

### Phase 4: Tensor Core Optimization (Week 3-4)

**Goal:** Use FP16/TF32 Tensor Cores for speed

**Option A: WMMA approach**
- [ ] Implement GEMM kernel using WMMA
- [ ] Replace cuBLAS calls with WMMA kernels
- [ ] Handle non-multiple-of-16 matrix sizes (padding)

**Option B: CUTLASS approach** (recommended if time permits)
- [ ] Set up CUTLASS in your project
- [ ] Use CUTLASS GEMM for matrix operations
- [ ] Configure for FP16/TF32 precision

**Deliverable:** Fast Tensor Core implementation

### Phase 5: Benchmarking & Analysis (Week 4-5)

**Goal:** Comprehensive performance evaluation

Tasks:
- [ ] Benchmark on various matrix sizes (512×512 to 16K×16K)
- [ ] Test on dense random matrices
- [ ] Test on structured matrices (2D Poisson)
- [ ] Vary condition numbers
- [ ] Profile with Nsight Compute
- [ ] Create plots: speedup vs. size, accuracy vs. iterations
- [ ] Analyze performance bottlenecks

**Deliverable:** Complete benchmark suite and results

### Phase 6: (Optional) Adaptive Precision (Week 5+)

**Goal:** Dynamic precision selection

Tasks:
- [ ] Monitor residual reduction rate
- [ ] Implement threshold-based precision switching
- [ ] Test on ill-conditioned matrices
- [ ] Compare against fixed precision

**Deliverable:** Adaptive solver

### Phase 7: Report & Presentation (Final Week)

- [ ] Write final report
- [ ] Create presentation slides
- [ ] Prepare demo for presentation
- [ ] Polish code and documentation

## Step 5: IBEX-Specific Tips

### Interactive Session (for testing):
```bash
srun --time=01:00:00 --gres=gpu:1 --mem=16G --pty bash
# Now you can run CUDA programs directly
./build/bin/my_program
```

### Batch Job (for benchmarking):
Create `job.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=mixed_precision_ir
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1  # or a100:1 if available
#SBATCH --mem=32G
#SBATCH --output=results/output_%j.txt

module load cuda/11.8
module load cmake/3.20

cd $SLURM_SUBMIT_DIR
./build/bin/benchmark_solver
```

Submit with: `sbatch job.slurm`

### Check GPU availability:
```bash
sinfo -p gpu  # See available GPU nodes
squeue -u $USER  # Check your jobs
```

## Step 6: Testing Strategy

### Unit Tests
1. Test on known matrices with known solutions
2. Test identity matrix (should converge in 1 iteration)
3. Test diagonal matrices (well-conditioned)

### Correctness Tests
```python
# Generate test case:
A = random_matrix(n, n)
x_true = random_vector(n)
b = A @ x_true

# Solve:
x_computed = solve(A, b)

# Check:
relative_error = ||x_computed - x_true|| / ||x_true||
assert relative_error < 1e-10  # for FP64 accuracy
```

### Convergence Tests
- Monitor: ||rₖ|| / ||r₀|| should decrease geometrically
- Expected: 2-3 iterations for κ(A) ~ 10²-10⁴

## Debugging Tips

1. **Start small:** Test on 4×4 matrices first, print everything
2. **Check dimensions:** Matrix dimension mismatches are common
3. **Verify precision conversions:** Print values before/after conversion
4. **Profile early:** Use `nvprof` or Nsight Compute to find bottlenecks
5. **Compare to CPU:** Implement simple CPU version for verification

## Resources Quick Links

- **CUTLASS:** https://github.com/NVIDIA/cutlass
- **cuBLAS:** https://docs.nvidia.com/cuda/cublas/
- **cuSOLVER:** https://docs.nvidia.com/cuda/cusolver/
- **WMMA:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
- **Nsight Compute:** https://developer.nvidia.com/nsight-compute

## Questions to Answer in Your Report

1. How does speedup scale with matrix size?
2. What's the crossover point where mixed-precision wins?
3. How does condition number affect convergence?
4. What's the memory bandwidth utilization?
5. Are you compute-bound or memory-bound?
6. How do different precision combinations perform?

Good luck with your project! 🚀

