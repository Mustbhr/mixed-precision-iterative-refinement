/**
 * Mixed-Precision Iterative Refinement Solver
 * CS380 Project - Mustafa Albahrani
 *
 * Phase 2: FP32 Factorization with FP64 Iterative Refinement
 *
 * ALGORITHM OVERVIEW:
 * ===================
 * 1. Factor A in FP32 (fast, ~2x speedup over FP64)
 * 2. Solve initial x₀ in FP32
 * 3. Refine iteratively:
 *    - r = b - Ax        (FP64: catches small errors)
 *    - Solve Ad = r      (FP32: reuse LU factors)
 *    - x = x + d         (FP64: accumulate accurately)
 *    - Check convergence
 *
 * RESULT: FP64 accuracy at nearly FP32 speed!
 */

#include "solver.h"
#include <cmath>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>

// Error checking macros
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at "       \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::cerr << "cuBLAS Error: " << status << " at " << __FILE__ << ":"     \
                << __LINE__ << std::endl;                                      \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define CUSOLVER_CHECK(call)                                                   \
  do {                                                                         \
    cusolverStatus_t status = call;                                            \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
      std::cerr << "cuSOLVER Error: " << status << " at " << __FILE__ << ":"   \
                << __LINE__ << std::endl;                                      \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// ============================================================================
// CUDA KERNEL: Convert FP64 to FP32
// ============================================================================
// Why we need this: cuSOLVER FP32 functions need float* input,
// but our matrices come in as double* for compatibility with baseline.
__global__ void convert_fp64_to_fp32_kernel(const double *src, float *dst,
                                            int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = static_cast<float>(src[idx]);
  }
}

// ============================================================================
// CUDA KERNEL: Convert FP32 to FP64
// ============================================================================
// Why we need this: After solving in FP32, we need to convert back to FP64
// for the high-precision residual computation and solution accumulation.
__global__ void convert_fp32_to_fp64_kernel(const float *src, double *dst,
                                            int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = static_cast<double>(src[idx]);
  }
}

// Helper function wrappers
void convert_fp64_to_fp32(const double *d_src, float *d_dst, int n) {
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  convert_fp64_to_fp32_kernel<<<numBlocks, blockSize>>>(d_src, d_dst, n);
}

void convert_fp32_to_fp64(const float *d_src, double *d_dst, int n) {
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  convert_fp32_to_fp64_kernel<<<numBlocks, blockSize>>>(d_src, d_dst, n);
}

// ============================================================================
// FP32 LU SOLVER (for comparison and as part of mixed-precision)
// ============================================================================
int solve_lu_fp32(const double *A_host, const double *b_host, double *x_host,
                  int n, double *time_ms) {
  cudaEvent_t start, stop;
  if (time_ms) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  }

  // Create handles
  cusolverDnHandle_t solver_handle;
  CUSOLVER_CHECK(cusolverDnCreate(&solver_handle));

  // Allocate device memory
  // We need both FP64 (for input) and FP32 (for solving)
  double *d_A_fp64, *d_b_fp64;
  float *d_A_fp32, *d_b_fp32;

  CUDA_CHECK(cudaMalloc(&d_A_fp64, n * n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_b_fp64, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_A_fp32, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b_fp32, n * sizeof(float)));

  // Convert from row-major to column-major (cuSOLVER requirement)
  double *A_col_major = new double[n * n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A_col_major[j * n + i] = A_host[i * n + j]; // Transpose
    }
  }

  // Copy to device
  CUDA_CHECK(cudaMemcpy(d_A_fp64, A_col_major, n * n * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_b_fp64, b_host, n * sizeof(double), cudaMemcpyHostToDevice));
  delete[] A_col_major;

  // Convert FP64 -> FP32 on GPU
  convert_fp64_to_fp32(d_A_fp64, d_A_fp32, n * n);
  convert_fp64_to_fp32(d_b_fp64, d_b_fp32, n);

  // Allocate workspace for FP32 LU
  int *d_ipiv, *d_info;
  CUDA_CHECK(cudaMalloc(&d_ipiv, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

  int lwork = 0;
  CUSOLVER_CHECK(
      cusolverDnSgetrf_bufferSize(solver_handle, n, n, d_A_fp32, n, &lwork));

  float *d_work;
  CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));

  // LU factorization in FP32
  CUSOLVER_CHECK(cusolverDnSgetrf(solver_handle, n, n, d_A_fp32, n, d_work,
                                  d_ipiv, d_info));

  // Check factorization success
  int info_h;
  CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (info_h != 0) {
    std::cerr << "FP32 LU factorization failed, info = " << info_h << std::endl;
    cudaFree(d_A_fp64);
    cudaFree(d_b_fp64);
    cudaFree(d_A_fp32);
    cudaFree(d_b_fp32);
    cudaFree(d_ipiv);
    cudaFree(d_info);
    cudaFree(d_work);
    cusolverDnDestroy(solver_handle);
    return -1;
  }

  // Solve in FP32
  CUSOLVER_CHECK(cusolverDnSgetrs(solver_handle, CUBLAS_OP_N, n, 1, d_A_fp32, n,
                                  d_ipiv, d_b_fp32, n, d_info));

  // Convert result back to FP64 and copy to host
  double *d_x_fp64;
  CUDA_CHECK(cudaMalloc(&d_x_fp64, n * sizeof(double)));
  convert_fp32_to_fp64(d_b_fp32, d_x_fp64, n);
  CUDA_CHECK(
      cudaMemcpy(x_host, d_x_fp64, n * sizeof(double), cudaMemcpyDeviceToHost));

  if (time_ms) {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms_float;
    cudaEventElapsedTime(&time_ms_float, start, stop);
    *time_ms = static_cast<double>(time_ms_float);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // Cleanup
  cudaFree(d_A_fp64);
  cudaFree(d_b_fp64);
  cudaFree(d_A_fp32);
  cudaFree(d_b_fp32);
  cudaFree(d_x_fp64);
  cudaFree(d_ipiv);
  cudaFree(d_info);
  cudaFree(d_work);
  cusolverDnDestroy(solver_handle);

  return 0;
}

// ============================================================================
// MIXED-PRECISION ITERATIVE REFINEMENT SOLVER
// ============================================================================
// This is the main algorithm that achieves FP64 accuracy at FP32 speed!
//
// INPUT:
//   A_host      - n×n matrix (row-major, FP64)
//   b_host      - n×1 RHS vector (FP64)
//   n           - system size
//   max_iter    - maximum refinement iterations (typically 10)
//   tolerance   - convergence tolerance (1e-12 for FP64-level)
//
// OUTPUT:
//   x_host      - n×1 solution vector (FP64)
//   iterations  - number of refinement iterations used
//   timing      - detailed timing breakdown
//
int solve_mixed_precision_ir(const double *A_host, const double *b_host,
                             double *x_host, int n, int max_iterations,
                             double tolerance, int *iterations_used,
                             MixedPrecisionTiming *timing) {
  cudaEvent_t start_total, stop_total;
  cudaEvent_t start_factor, stop_factor;
  cudaEvent_t start_refine, stop_refine;

  cudaEventCreate(&start_total);
  cudaEventCreate(&stop_total);
  cudaEventCreate(&start_factor);
  cudaEventCreate(&stop_factor);
  cudaEventCreate(&start_refine);
  cudaEventCreate(&stop_refine);

  cudaEventRecord(start_total);

  // Create handles
  cusolverDnHandle_t solver_handle;
  cublasHandle_t blas_handle;
  CUSOLVER_CHECK(cusolverDnCreate(&solver_handle));
  CUBLAS_CHECK(cublasCreate(&blas_handle));

  // ========================================================================
  // MEMORY ALLOCATION
  // ========================================================================
  // We need multiple precision copies:
  // - A_fp64: for accurate residual computation (r = b - A*x)
  // - A_fp32: for fast LU factorization
  // - x_fp64: working solution (accumulated in FP64 for accuracy)
  // - r_fp64: residual vector (must be FP64 to catch small errors)
  // - d_fp32: correction vector (solved in FP32, then converted)

  double *d_A_fp64, *d_b_fp64, *d_x_fp64, *d_r_fp64;
  float *d_A_fp32, *d_r_fp32;
  double *d_d_fp64; // correction in FP64 (after conversion)

  CUDA_CHECK(cudaMalloc(&d_A_fp64, n * n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_b_fp64, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_x_fp64, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_r_fp64, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_d_fp64, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_A_fp32, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r_fp32, n * sizeof(float)));

  // Copy data to device (Row Major)
  double *d_A_row;
  CUDA_CHECK(cudaMalloc(&d_A_row, n * n * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_A_row, A_host, n * n * sizeof(double),
                        cudaMemcpyHostToDevice));

  // Transpose to Col Major using cuBLAS Geam
  double alpha_t = 1.0;
  double beta_t = 0.0;
  CUBLAS_CHECK(cublasDgeam(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n,
                           &alpha_t, d_A_row, n, &beta_t, d_A_row, n, d_A_fp64,
                           n));

  CUDA_CHECK(cudaFree(d_A_row));

  CUDA_CHECK(
      cudaMemcpy(d_b_fp64, b_host, n * sizeof(double), cudaMemcpyHostToDevice));

  // Convert A to FP32 for factorization
  convert_fp64_to_fp32(d_A_fp64, d_A_fp32, n * n);

  // ========================================================================
  // STEP 1: LU FACTORIZATION IN FP32 (the expensive part, but faster in FP32)
  // ========================================================================
  cudaEventRecord(start_factor);

  int *d_ipiv, *d_info;
  CUDA_CHECK(cudaMalloc(&d_ipiv, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

  int lwork = 0;
  CUSOLVER_CHECK(
      cusolverDnSgetrf_bufferSize(solver_handle, n, n, d_A_fp32, n, &lwork));

  float *d_work;
  CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));

  // Perform LU factorization: A_fp32 = P * L * U
  CUSOLVER_CHECK(cusolverDnSgetrf(solver_handle, n, n, d_A_fp32, n, d_work,
                                  d_ipiv, d_info));

  // Check success
  int info_h;
  CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (info_h != 0) {
    std::cerr << "FP32 LU factorization failed, info = " << info_h << std::endl;
    // Cleanup and return error
    cudaFree(d_A_fp64);
    cudaFree(d_b_fp64);
    cudaFree(d_x_fp64);
    cudaFree(d_r_fp64);
    cudaFree(d_d_fp64);
    cudaFree(d_A_fp32);
    cudaFree(d_r_fp32);
    cudaFree(d_ipiv);
    cudaFree(d_info);
    cudaFree(d_work);
    cusolverDnDestroy(solver_handle);
    cublasDestroy(blas_handle);
    return -1;
  }

  cudaEventRecord(stop_factor);

  // ========================================================================
  // STEP 2: INITIAL SOLVE IN FP32
  // ========================================================================
  // Solve Ax₀ = b using the FP32 LU factors
  // This gives us a starting point for refinement

  // Copy b to a temp FP32 buffer for the initial solve
  float *d_x_fp32_temp;
  CUDA_CHECK(cudaMalloc(&d_x_fp32_temp, n * sizeof(float)));
  convert_fp64_to_fp32(d_b_fp64, d_x_fp32_temp, n);

  CUSOLVER_CHECK(cusolverDnSgetrs(solver_handle, CUBLAS_OP_N, n, 1, d_A_fp32, n,
                                  d_ipiv, d_x_fp32_temp, n, d_info));

  // Convert initial solution to FP64
  convert_fp32_to_fp64(d_x_fp32_temp, d_x_fp64, n);
  cudaFree(d_x_fp32_temp);

  // ========================================================================
  // STEP 3: ITERATIVE REFINEMENT LOOP
  // ========================================================================
  // This is where the magic happens!
  // Each iteration:
  //   1. Compute residual r = b - Ax (in FP64 - catches small errors)
  //   2. Solve Ad = r for correction d (in FP32 - reuse LU factors)
  //   3. Update x = x + d (in FP64 - accumulate accurately)
  //   4. Check if ||r|| is small enough

  cudaEventRecord(start_refine);

  double alpha, beta;
  double residual_norm, b_norm;
  int iter;

  // Compute ||b|| for relative residual
  CUBLAS_CHECK(cublasDnrm2(blas_handle, n, d_b_fp64, 1, &b_norm));

  for (iter = 0; iter < max_iterations; iter++) {
    // ---------------------------------------------------------------------
    // STEP 3a: Compute residual r = b - A*x (in FP64)
    // ---------------------------------------------------------------------
    // Why FP64? The residual measures how far off our solution is.
    // Small errors (1e-14) would be rounded away in FP32.
    // By computing in FP64, we can detect and correct these tiny errors.

    // r = b (copy b to r)
    CUBLAS_CHECK(cublasDcopy(blas_handle, n, d_b_fp64, 1, d_r_fp64, 1));

    // r = r - A*x = b - A*x
    // Using DGEMV: y = alpha*A*x + beta*y
    // We want: r = -1.0 * A * x + 1.0 * r
    alpha = -1.0;
    beta = 1.0;
    CUBLAS_CHECK(cublasDgemv(blas_handle, CUBLAS_OP_N, n, n, &alpha, d_A_fp64,
                             n, d_x_fp64, 1, &beta, d_r_fp64, 1));

    // Check convergence: ||r|| / ||b||
    CUBLAS_CHECK(cublasDnrm2(blas_handle, n, d_r_fp64, 1, &residual_norm));
    double relative_residual = residual_norm / b_norm;

    if (relative_residual < tolerance) {
      iter++; // Count this iteration
      break;  // Converged!
    }

    // ---------------------------------------------------------------------
    // STEP 3b: Solve Ad = r for correction (in FP32, reuse LU factors)
    // ---------------------------------------------------------------------
    // Why FP32? We already have the LU factors in FP32.
    // The correction doesn't need to be exact - just good enough.
    // We'll correct any remaining error in the next iteration.

    // Convert residual to FP32
    convert_fp64_to_fp32(d_r_fp64, d_r_fp32, n);

    // Solve using existing LU factors: d = L\(U\r)
    CUSOLVER_CHECK(cusolverDnSgetrs(solver_handle, CUBLAS_OP_N, n, 1, d_A_fp32,
                                    n, d_ipiv, d_r_fp32, n, d_info));

    // Convert correction back to FP64
    convert_fp32_to_fp64(d_r_fp32, d_d_fp64, n);

    // ---------------------------------------------------------------------
    // STEP 3c: Update solution x = x + d (in FP64)
    // ---------------------------------------------------------------------
    // Why FP64? We're accumulating improvements to the solution.
    // Using FP64 ensures we don't lose precision as we refine.

    alpha = 1.0;
    CUBLAS_CHECK(cublasDaxpy(blas_handle, n, &alpha, d_d_fp64, 1, d_x_fp64, 1));
  }

  cudaEventRecord(stop_refine);
  cudaEventRecord(stop_total);
  cudaEventSynchronize(stop_total);

  // ========================================================================
  // COPY RESULT AND RECORD TIMING
  // ========================================================================
  CUDA_CHECK(
      cudaMemcpy(x_host, d_x_fp64, n * sizeof(double), cudaMemcpyDeviceToHost));

  if (iterations_used) {
    *iterations_used = iter;
  }

  if (timing) {
    float factor_ms, refine_ms, total_ms;
    cudaEventElapsedTime(&factor_ms, start_factor, stop_factor);
    cudaEventElapsedTime(&refine_ms, start_refine, stop_refine);
    cudaEventElapsedTime(&total_ms, start_total, stop_total);

    timing->factorization_ms = factor_ms;
    timing->refinement_ms = refine_ms;
    timing->total_ms = total_ms;
    timing->iterations = iter;
    timing->final_residual = residual_norm / b_norm;
  }

  // Cleanup
  cudaFree(d_A_fp64);
  cudaFree(d_b_fp64);
  cudaFree(d_x_fp64);
  cudaFree(d_r_fp64);
  cudaFree(d_d_fp64);
  cudaFree(d_A_fp32);
  cudaFree(d_r_fp32);
  cudaFree(d_ipiv);
  cudaFree(d_info);
  cudaFree(d_work);

  cudaEventDestroy(start_total);
  cudaEventDestroy(stop_total);
  cudaEventDestroy(start_factor);
  cudaEventDestroy(stop_factor);
  cudaEventDestroy(start_refine);
  cudaEventDestroy(stop_refine);

  cusolverDnDestroy(solver_handle);
  cublasDestroy(blas_handle);

  return 0;
}
