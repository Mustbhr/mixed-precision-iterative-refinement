# Mixed-Precision Iterative Refinement on GPUs

**CS380 - GPU and GPGPU Programming**  
**Student:** Mustafa Albahrani  
**Instructor:** Prof. Markus Hadwiger  
**TA:** Peter Rautek

## Project Overview

This project implements mixed-precision iterative refinement (IR) solvers for linear systems on GPUs, leveraging Tensor Cores for high-performance low-precision arithmetic while achieving FP64-level accuracy.

### Key Features
- Mixed-precision solver using FP16/TF32 for factorizations
- FP32/FP64 for residual corrections
- CUDA C++ implementation with CUTLASS/WMMA API
- Benchmarking on dense and structured matrices
- Performance-accuracy trade-off analysis

## Background

### What is Iterative Refinement?

Iterative refinement is a technique to improve the accuracy of solutions to linear systems Ax = b:
1. Compute an approximate solution x₀ using low-precision factorization
2. Calculate residual: r = b - Ax₀ (in higher precision)
3. Solve correction: Ad = r (using low-precision factorization)
4. Update solution: x₁ = x₀ + d
5. Repeat until desired accuracy is reached

### Why Mixed-Precision?

Modern GPUs (especially with Tensor Cores) are much faster at FP16/TF32 arithmetic than FP64:
- **Speed:** ~10-20x faster for low-precision operations
- **Memory:** Reduced bandwidth requirements
- **Accuracy:** Can still achieve FP64-level accuracy through iterative refinement

## Project Structure

```
.
├── src/               # Source code
│   ├── solvers/      # Mixed-precision IR solver implementations
│   ├── kernels/      # CUDA kernels (CUTLASS/WMMA)
│   ├── utils/        # Helper functions, matrix generators
│   └── benchmarks/   # Benchmark and test code
├── include/          # Header files
├── tests/            # Test matrices and validation
├── results/          # Benchmark results and plots
├── docs/             # Documentation and notes
├── CMakeLists.txt    # Build configuration
└── README.md         # This file
```

## Building and Running

### On MacOS (Development)


### On IBEX (Deployment)
```bash
# Load required modules
module load cuda/11.x
module load cmake/3.x

# Build
mkdir build && cd build
cmake ..
make

# Run benchmarks
./bin/benchmark_solver
```

## Implementation Roadmap

### Phase 1: Baseline Implementation
- [ ] Basic CUDA project setup
- [ ] Matrix generation utilities (dense and structured)
- [ ] LU/Cholesky factorization using cuBLAS
- [ ] Basic iterative refinement loop
- [ ] Residual computation in mixed precision

### Phase 2: Tensor Core Optimization
- [ ] Implement WMMA/CUTLASS kernels for matrix operations
- [ ] FP16/TF32 factorization
- [ ] FP32/FP64 residual correction
- [ ] Performance profiling with Nsight Compute

### Phase 3: Benchmarking
- [ ] Test on dense matrices
- [ ] Test on structured matrices (2D Poisson)
- [ ] Compare against pure FP64 cuBLAS
- [ ] Convergence analysis
- [ ] Performance-accuracy trade-offs

### Phase 4: (Optional) Adaptive Precision
- [ ] Implement threshold-based precision switching
- [ ] Monitor residual norms
- [ ] Dynamic precision selection

## References

- Haidar et al., "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers," SC18
- NVIDIA CUTLASS: https://github.com/NVIDIA/cutlass
- CUDA WMMA API Documentation
- cuBLAS Library Documentation

## Progress Log

- **[date: 24-11-2025]** - Project initialization and repository setup

