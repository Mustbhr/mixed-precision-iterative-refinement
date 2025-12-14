# Accelerated Mixed-Precision Iterative Refinement on NVIDIA A100

**CS380 - GPU and GPGPU Programming**  
**Student:** Mustafa Albahrani  
**Instructor:** Prof. Markus Hadwiger  

## Project Overview
This project implements a high-performance **Linear System Solver (Ax=b)** that leverages the specialized **Tensor Cores** of the NVIDIA A100 GPU to achieve unprecedented speedups while maintaining full double-precision (FP64) accuracy.

By using **Iterative Refinement**, we combine the raw throughput of **FP16 Tensor Cores** (312 TFLOPS) with the precision of standard **FP64 Cores**:
1.  **Factorization**: Decompose the matrix $A$ using low-precision (FP16/TF32) for speed.
2.  **Refinement**: Correct the error using high-precision (FP64) residuals.

### Key Achievements
-   **Speedup**: Achieved **6.40x faster** performance than optimized cuSOLVER FP64 routines.
-   **Hardware Utilization**: Successfully targeted **Ampere Tensor Cores** (Verified via Nsight Compute).
-   **Algorithm**: Implemented a custom **Block LU Factorization** with Global Pivoting to handle numerical instability in half-precision.

## Performance Results (NVIDIA A100)
Benchmark results for Solving $Ax=b$ (Random Dense Matrix):

| Matrix Size | FP64 Baseline (ms) | Mixed-Precision (ms) | Speedup | Relative Residual Error |
| :--- | ---: | ---: | ---: | :--- |
| **8192** | 1002.11 | **169.85** | **5.90x** | $4.9 \times 10^{-15}$ |
| **16000** | 2294.46 | **507.15** | **4.52x** | $7.0 \times 10^{-15}$ |
| **32000** | 10342.19 | **1625.74** | **6.36x** | $9.8 \times 10^{-15}$ |
| **40000** | 15936.95 | **2490.96** | **6.40x** | $1.0 \times 10^{-14}$ |

## Implementation Details
The project evolved through three phases of optimization:

### Phase 1: FP64 Baseline
Standard LU factorization using `cusolverDnDgetrf`. Reliable but limited by FP64 throughput (19.5 TFLOPS).

### Phase 2: Mixed-Precision (TF32)
Used `cusolverDnSgetrf` (FP32). On the A100, this implicitly uses **TF32 Tensor Cores** (156 TFLOPS), achieving a ~3x speedup.

### Phase 3: Tensor Core Acceleration (FP16)
The final custom implementation:
-   **Algorithm**: Right-Looking Blocked LU Decomposition.
-   **Compute**: Explicitly uses `cublasGemmEx` with `CUDA_R_16F` inputs and `CUBLAS_TENSOR_OP_MATH` mode to force FP16 Tensor Core usage.
-   **Stability**: Implements "Tall Panel" factorization with `cusolverDnSlaswp` (pivoting) on the A100 to prevent NaN/Inf explosions common in FP16.
-   **Memory**: Uses customized coalesced layout formatting kernels.

## Building and Running

### Prerequisites
-   NVIDIA GPU (Volt/Ampere/Hopper recommended)
-   CUDA Toolkit 11.0+
-   CMake 3.18+

### Build Instructions
```bash
mkdir build && cd build
cmake ..
make -j
```

### Running the Benchmark
```bash
# Run the solver comparison
./bin/mixed_precision_ir

# Run with profiling logic (single targeted run)
./bin/mixed_precision_ir --profile
```

### Profiling
Scripts for Nsight Systems (`nsys`) and Nsight Compute (`ncu`) are available in `profile_job.slurm`.

## References
1.  Haidar et al., *"Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers,"* SC18.
2.  NVIDIA A100 Tensor Core Architecture Whitepaper.
3.  Higham, N. J., *"Accuracy and Stability of Numerical Algorithms."*
