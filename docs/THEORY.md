# Mixed-Precision Iterative Refinement - Theory and Background

## Table of Contents
1. [The Problem: Solving Linear Systems](#the-problem)
2. [Precision Formats on GPUs](#precision-formats)
3. [Iterative Refinement Algorithm](#iterative-refinement)
4. [Tensor Cores](#tensor-cores)
5. [Implementation Strategy](#implementation-strategy)

## The Problem: Solving Linear Systems

We want to solve **Ax = b** where:
- A is an n×n matrix (dense or structured)
- b is the right-hand side vector
- x is the unknown solution

Traditional approach:
1. Compute LU or Cholesky factorization: **A = LU**
2. Solve **Ly = b** (forward substitution)
3. Solve **Ux = y** (backward substitution)

**Challenge:** In FP64, this is accurate but slow on GPUs. In FP16, it's fast but inaccurate.

## Precision Formats on GPUs

### Float16 (FP16) - Half Precision
- **Size:** 16 bits (1 sign, 5 exponent, 10 mantissa)
- **Range:** ±65,504
- **Precision:** ~3-4 decimal digits
- **GPU Speed:** Very fast on Tensor Cores (up to 312 TFLOPS on A100)

### TensorFloat-32 (TF32)
- **Size:** 32 bits storage, but uses 19 bits for computation
- **Range:** Same as FP32
- **Precision:** ~3 decimal digits (like FP16 mantissa)
- **GPU Speed:** Fast on Ampere+ Tensor Cores (156 TFLOPS on A100)
- **Note:** Transparent replacement for FP32 on Ampere GPUs

### Float32 (FP32) - Single Precision
- **Size:** 32 bits (1 sign, 8 exponent, 23 mantissa)
- **Range:** ±3.4×10³⁸
- **Precision:** ~7 decimal digits
- **GPU Speed:** Fast (19.5 TFLOPS on A100)

### Float64 (FP64) - Double Precision
- **Size:** 64 bits (1 sign, 11 exponent, 52 mantissa)
- **Range:** ±1.8×10³⁰⁸
- **Precision:** ~15-16 decimal digits
- **GPU Speed:** Slow (9.7 TFLOPS on A100, consumer GPUs even slower)

### Speed Comparison (NVIDIA A100)
```
FP16 Tensor Core:  312 TFLOPS  (32x faster than FP64)
TF32 Tensor Core:  156 TFLOPS  (16x faster than FP64)
FP32:               19.5 TFLOPS (2x faster than FP64)
FP64:                9.7 TFLOPS (baseline)
```

## Iterative Refinement Algorithm

### Classic Iterative Refinement (Wilkinson, 1963)

The key insight: Even if you solve Ax = b approximately, you can improve it!

**Algorithm:**
```
1. Solve Ax₀ = b approximately (using LU factorization in low precision)
2. for k = 0, 1, 2, ... until convergence:
   a. Compute residual: rₖ = b - Axₖ    [in higher precision]
   b. Solve correction: Adₖ = rₖ        [reuse LU factors]
   c. Update solution: xₖ₊₁ = xₖ + dₖ   [in higher precision]
   d. Check convergence: ||rₖ|| < tolerance
```

### Why Does This Work?

**Mathematical Intuition:**
- The error in our solution is eₖ = x* - xₖ (where x* is the true solution)
- We have: A(x* - xₖ) = b - Axₖ = rₖ
- So: Aeₖ = rₖ
- By solving Adₖ = rₖ, we get an approximation of the error!
- Adding dₖ to xₖ reduces the error

**Key Points:**
1. The residual **rₖ = b - Axₖ** must be computed in higher precision
2. The correction step reuses the LU factorization (cheap!)
3. Each iteration improves accuracy by roughly one precision level
4. Typically converges in 2-3 iterations for well-conditioned systems

### Mixed-Precision Variant

**Modern GPU Approach (Haidar et al., SC'18):**

```
Factorization:     Use FP16/TF32 (fast on Tensor Cores)
Residual:          Use FP32 or FP64 (accurate)
Correction:        Use FP16/TF32 (reuse factorization)
Solution update:   Use FP32 or FP64 (accumulate accurately)
```

**Precision Choices:**
- **Aggressive:** FP16 factorization, FP32 residual → FP32 accuracy
- **Moderate:** TF32 factorization, FP64 residual → FP64 accuracy
- **Conservative:** FP32 factorization, FP64 residual → FP64 accuracy

## Tensor Cores

### What Are Tensor Cores?

Special hardware units on NVIDIA GPUs (Volta, Turing, Ampere, Hopper) designed for mixed-precision matrix operations.

**Basic Operation:** D = A × B + C
- A: m×k matrix (FP16 or TF32)
- B: k×n matrix (FP16 or TF32)
- C: m×n matrix (FP32)
- D: m×n matrix (FP32)
- Accumulation in FP32 for better accuracy

### Tensor Core Programming

**Two APIs:**

#### 1. WMMA (Warp Matrix Multiply-Accumulate)
```cuda
// Fragment types hold tile data
wmma::fragment<wmma::matrix_a, M, N, K, half> a_frag;
wmma::fragment<wmma::matrix_b, M, N, K, half> b_frag;
wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

// Load from memory
wmma::load_matrix_sync(a_frag, A, lda);
wmma::load_matrix_sync(b_frag, B, ldb);
wmma::fill_fragment(c_frag, 0.0f);

// Multiply-accumulate
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// Store result
wmma::store_matrix_sync(C, c_frag, ldc, wmma::mem_row_major);
```

**Pros:** Simple, official NVIDIA API  
**Cons:** Limited flexibility, fixed tile sizes (16×16×16)

#### 2. CUTLASS
```cpp
// Define operation
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,     // Element type A
    cutlass::layout::RowMajor,
    cutlass::half_t,     // Element type B
    cutlass::layout::ColumnMajor,
    float,               // Element type C
    cutlass::layout::RowMajor,
    float                // Accumulator type
>;

// Launch kernel
Gemm gemm_op;
gemm_op({m, n, k}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});
```

**Pros:** Highly optimized, flexible, template-based  
**Cons:** Steeper learning curve, complex templates

## Implementation Strategy

### Phase 1: Baseline with cuBLAS

Start simple before optimizing:

```cuda
// 1. Factorization (FP32)
cublasSgetrf(handle, n, A_fp32, lda, ipiv, info);

// 2. Initial solve (FP32)
cublasSgetrs(handle, CUBLAS_OP_N, n, 1, A_fp32, lda, ipiv, x_fp32, n);

// 3. Iterative refinement loop
for (int iter = 0; iter < max_iter; iter++) {
    // Compute residual: r = b - Ax (in FP64)
    cublasDcopy(handle, n, b_fp64, 1, r_fp64, 1);           // r = b
    cublasDgemv(handle, CUBLAS_OP_N, n, n, -1.0,            // r = r - Ax
                A_fp64, lda, x_fp64, 1, 1.0, r_fp64, 1);
    
    // Solve correction: Ad = r (in FP32, reuse factorization)
    convert_fp64_to_fp32(r_fp64, r_fp32, n);
    cublasSgetrs(handle, CUBLAS_OP_N, n, 1, A_fp32, lda, ipiv, d_fp32, n);
    
    // Update solution: x = x + d (in FP64)
    convert_fp32_to_fp64(d_fp32, d_fp64, n);
    cublasDaxpy(handle, n, 1.0, d_fp64, 1, x_fp64, 1);
    
    // Check convergence
    double residual_norm = compute_norm(r_fp64, n);
    if (residual_norm < tolerance) break;
}
```

### Phase 2: Tensor Core Optimization

Replace factorization and triangular solves with Tensor Core kernels:

```cuda
// Use CUTLASS or WMMA for:
// 1. LU factorization in FP16/TF32
// 2. Triangular solves in FP16/TF32
// Keep residual computation in FP64
```

### Phase 3: Performance Analysis

**Metrics to measure:**
1. **Time:** Total solve time, time per iteration
2. **Accuracy:** Relative error ||x_computed - x_true|| / ||x_true||
3. **Convergence:** Number of iterations to reach tolerance
4. **Condition number:** How does κ(A) affect performance?

**Test matrices:**
1. Random dense matrices (various sizes: 1K×1K to 16K×16K)
2. Structured matrices (2D Poisson: -Δu = f on grid)
3. Varying condition numbers (well-conditioned to ill-conditioned)

### Phase 4: (Optional) Adaptive Precision

Monitor convergence and switch precisions:

```cuda
if (residual_reduction < 0.1) {
    // Slow convergence, increase precision
    factorization_precision = FP32;
} else if (residual_reduction > 0.5) {
    // Fast convergence, can use lower precision
    factorization_precision = FP16;
}
```

## Key Implementation Challenges

1. **Memory management:** Need multiple precision copies of matrices
2. **Precision conversion:** Efficient FP16 ↔ FP32 ↔ FP64 conversions
3. **Numerical stability:** Some matrices won't converge with low-precision factorization
4. **Performance:** Memory bandwidth vs. compute trade-offs

## Expected Results

**For well-conditioned dense matrices:**
- **Speedup:** 3-10x faster than pure FP64
- **Accuracy:** Same as FP64 (10⁻¹⁵ relative error)
- **Iterations:** 2-3 iterations typically

**For ill-conditioned matrices:**
- May need more iterations or fail to converge
- Adaptive precision can help

## References and Resources

### Papers
- Haidar et al., "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers," SC'18
- Higham, "Iterative Refinement for Linear Systems and LAPACK," IMA Journal of Numerical Analysis, 1997

### NVIDIA Documentation
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [WMMA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [cuBLAS Library](https://docs.nvidia.com/cuda/cublas/)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute)

### Tutorials
- [CUTLASS Quickstart](https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md)
- [Tensor Core Programming](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)

