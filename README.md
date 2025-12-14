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

### Phase 1: Baseline Implementation (Complete)
- [x] Basic CUDA project setup
- [x] Matrix generation utilities (dense and structured)
- [x] FP64 LU factorization using cuSOLVER (Baseline)
- [x] Basic iterative refinement loop validation

### Phase 2: Mixed-Precision IR - Standard (Complete)
- [x] Implement Standard Mixed-Precision Solver (FP32/TF32 factorization)
- [x] Integrate FP64 residual correction
- [x] **Analysis**: On A100, this uses TF32 Tensor Cores (156 TFLOPS) via standard cuSOLVER
- [x] Achieve ~3x speedup over Baseline

### Phase 3: Tensor Core Acceleration - Manual (Complete)
- [x] Implement **manual** Block LU solver
- [ ] Implement WMMA/CUTLASS kernels (Replaced by `cublasGemmEx` + `CUDA_R_16F` for simplicity and speed)
- [x] Explicitly target **FP16 Tensor Cores** (312 TFLOPS)
- [x] Implement "Tall Panel" factorization with Global Pivoting for stability
- [x] Achieve **~8x-16x speedup** over Baseline
- [x] **Benchmarking & Analysis**:
    - [x] Compare against pure FP64 and Phase 2
    - [x] Convergence analysis (Residual logging)
    - [x] Performance-accuracy trade-offs

## Convergence Analysis
The solver demonstrates theoretical iterative refinement behavior:
- **Initial Guess**: Forward error $\approx 10^{-4}$ (limited by FP16/TF32 precision).
- **Iteration 1**: Residual drops to $\approx 10^{-10}$.
- **Iteration 2**: Residual drops to $\approx 10^{-15}$ (full FP64 accuracy).
- **Conclusion**: The mixed-precision strategy successfully recovers double-precision accuracy from low-precision factorizations.

## References

- Haidar et al., "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers," SC18
- NVIDIA CUTLASS: https://github.com/NVIDIA/cutlass
- CUDA WMMA API Documentation
- cuBLAS Library Documentation

## Progress Log

- **[24-11-2025]** - Project initialization and repository setup
- **[27-11-2025]** - Phase 1 Completed. i.e. FP64 LU factorization (baseline). 

