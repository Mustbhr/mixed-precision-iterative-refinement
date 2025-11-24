# Quick Start Guide

## 🚀 Immediate Next Steps

### 1. Create GitHub Repository (5 minutes)

1. Go to https://github.com/new
2. Repository name: `mixed-precision-iterative-refinement`
3. Description: "Mixed-Precision Iterative Refinement Solver using GPU Tensor Cores - CS380 Project"
4. **Public** repository
5. **Do NOT** initialize with README
6. Click "Create repository"

### 2. Push Your Code to GitHub

```bash
cd /Users/mustafa/Desktop/KAUST/CS380/CS380-Project

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/mixed-precision-iterative-refinement.git

# Push
git push -u origin main
```

### 3. Read the Documentation

I've created comprehensive guides for you:

📚 **Start here:** `docs/THEORY.md`
- Explains mixed-precision iterative refinement
- Covers precision formats (FP16, TF32, FP32, FP64)
- Describes Tensor Cores and why they're fast
- ~30 minutes to read

📖 **Then read:** `docs/GETTING_STARTED.md`
- Complete learning path
- Week-by-week implementation roadmap
- Testing strategies
- Resources and links

🖥️ **For IBEX:** `docs/IBEX_GUIDE.md`
- How to connect and use KAUST's HPC cluster
- GPU job submission
- File transfer
- Troubleshooting

### 4. Test on IBEX (Tomorrow or when ready)

```bash
# SSH to IBEX
ssh YOUR_KAUST_ID@ilogin.ibex.kaust.edu.sa

# Clone your repository
git clone https://github.com/YOUR_USERNAME/mixed-precision-iterative-refinement.git
cd mixed-precision-iterative-refinement

# Load modules
module load cuda/11.8 cmake/3.20

# Build
mkdir build && cd build
cmake ..
make

# Test in interactive session
srun --time=00:30:00 --gres=gpu:v100:1 --pty bash
./bin/mixed_precision_ir
exit
```

## 📋 Project Structure

```
CS380-Project/
├── README.md                    # Main project overview
├── QUICKSTART.md               # This file
├── CMakeLists.txt              # Build configuration
│
├── docs/
│   ├── THEORY.md               # 📚 START HERE: Theory & background
│   ├── GETTING_STARTED.md      # Learning path & roadmap
│   └── IBEX_GUIDE.md          # HPC cluster guide
│
├── src/
│   ├── main.cu                 # Starter code (Hello CUDA + cuBLAS test)
│   ├── CMakeLists.txt         # Source build config
│   ├── solvers/               # (empty) Future: IR solver implementations
│   ├── kernels/               # (empty) Future: Tensor Core kernels
│   ├── utils/                 # (empty) Future: Matrix generators, helpers
│   └── benchmarks/            # (empty) Future: Benchmark code
│
├── include/                    # (empty) Future: Header files
├── tests/                      # (empty) Future: Test matrices
└── results/                    # (empty) Benchmark outputs go here
```

## 🎯 Your Path Forward

### Week 1: Learning & Setup
- [ ] Push code to GitHub ← **Do this today!**
- [ ] Read `docs/THEORY.md` thoroughly
- [ ] Review CUDA basics if needed
- [ ] Study Haidar et al. SC'18 paper
- [ ] Test compilation on IBEX

### Week 2: Baseline Implementation
- [ ] Implement matrix generation utilities
- [ ] Create pure FP64 solver using cuBLAS/cuSOLVER
- [ ] Test on small matrices
- [ ] Verify correctness

### Week 3: Mixed-Precision Core
- [ ] Implement iterative refinement loop
- [ ] Add FP32 factorization, FP64 residual
- [ ] Test convergence
- [ ] Compare vs. pure FP64

### Week 4: Tensor Core Optimization
- [ ] Learn WMMA or CUTLASS
- [ ] Implement FP16/TF32 kernels
- [ ] Integrate into solver
- [ ] Profile with Nsight Compute

### Week 5: Benchmarking
- [ ] Run comprehensive benchmarks
- [ ] Test various matrix sizes and types
- [ ] Generate plots
- [ ] Analyze performance

### Final Week: Report & Presentation
- [ ] Write final report
- [ ] Create presentation
- [ ] Polish code
- [ ] Prepare demo

## 💡 Key Concepts to Understand

### 1. The Problem
Solve **Ax = b** where A is a large matrix
- Traditional: Use FP64 (slow but accurate)
- Our approach: Use FP16 for speed, achieve FP64 accuracy via iteration

### 2. The Algorithm
```
1. Solve Ax₀ = b in low precision (fast, approximate)
2. Loop:
   - Compute residual: r = b - Ax (high precision)
   - Solve correction: Ad = r (low precision, reuse factorization)
   - Update: x = x + d (high precision)
   - Check convergence
```

### 3. Why It Works
- Each iteration refines the solution
- High-precision residual catches errors
- Typically converges in 2-3 iterations
- Result: FP64 accuracy at FP16 speed!

### 4. Tensor Cores
- Special GPU hardware for matrix operations
- 10-20x faster than regular FP64 cores
- Support FP16, TF32 on modern GPUs
- We'll use them for the "fast" parts

## 📚 Essential Reading

### Must Read (This Week)
1. **docs/THEORY.md** - My comprehensive theory guide
2. **Haidar et al., SC'18** - The paper your project is based on
3. **NVIDIA Tensor Core blog** - Understanding the hardware

### Good to Read (When Implementing)
1. **cuBLAS documentation** - For baseline implementation
2. **WMMA documentation** - For Tensor Core programming
3. **CUTLASS examples** - For advanced optimization

## 🔗 Quick Links

- **Your Project:** https://github.com/YOUR_USERNAME/mixed-precision-iterative-refinement (create this!)
- **CUTLASS:** https://github.com/NVIDIA/cutlass
- **CUDA Docs:** https://docs.nvidia.com/cuda/
- **IBEX Docs:** https://www.hpc.kaust.edu.sa/ibex

## ❓ Questions to Keep in Mind

As you work on the project, think about:
1. How does matrix size affect the speedup?
2. When is mixed-precision faster than pure FP64?
3. How does matrix condition number affect convergence?
4. What's the memory vs. compute trade-off?
5. Are we limited by bandwidth or computation?

## 🆘 Getting Help

- **Theory questions:** Read docs/THEORY.md, ask instructor/TA
- **CUDA issues:** NVIDIA documentation, Stack Overflow
- **IBEX issues:** ibex@kaust.edu.sa
- **Git issues:** GitHub documentation

## ✅ Success Metrics

Your project is successful if you:
1. ✓ Implement working mixed-precision IR solver
2. ✓ Achieve FP64-level accuracy
3. ✓ Demonstrate speedup vs. pure FP64
4. ✓ Benchmark on various matrix types
5. ✓ Analyze and explain performance-accuracy trade-offs
6. ✓ (Bonus) Implement adaptive precision control

---

**Remember:** This is a substantial project. Take it step by step, test frequently, and don't hesitate to start simple before optimizing. Good luck! 🚀

