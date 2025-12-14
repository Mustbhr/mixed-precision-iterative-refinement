#include "solver.h"
#include <cmath>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iomanip>
#include <iostream>
#include <vector>

// ========================================================================================
// ERROR CHECKING MACROS
// ========================================================================================
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

// ========================================================================================
// KERNELS for Precision Conversion (Strided)
// ========================================================================================

/**
 * Cast FP32 -> FP16 with Stride
 * Extracts a submatrix from a larger FP32 matrix and converts to a contiguous
 * FP16 block.
 *
 * src: Pointer to the top-left of the submatrix in the large FP32 matrix
 * lda: Leading Dimension of src (usually n)
 * dst: Pointer to the contiguous FP16 destination buffer
 * rows: Number of rows in submatrix
 * cols: Number of cols in submatrix
 */
__global__ void cast_fp32_to_fp16_strided_kernel(const float *src, int lda,
                                                 __half *dst, int rows,
                                                 int cols) {
  // Optimized for Coalesced Access:
  // x-dimension (warp) should iterate over ROWS (contiguous in Col-Major)
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y * blockDim.y + threadIdx.y;

  if (r < rows && c < cols) {
    float val = src[c * lda + r];
    dst[c * rows + r] = __float2half(val);
  }
}

/**
 * Apply Pivots (LASWP equivalent)
 * Swaps rows in the right panel based on ipiv buffer.
 * A is the matrix pointer to the START of the panel (k, k+B)
 * ipiv is the pivot array for the current block (length B)
 * B is the block size
 * n is the leading dimension (stride)
 * width is the width of the panel (remaining columns)
 */
__global__ void apply_pivots_kernel(float *A, int lda, const int *ipiv, int B,
                                    int width) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= width)
    return;

  // Each thread handles one column of the panel
  // We iterate through the B pivots
  for (int i = 0; i < B; ++i) {
    int pivot_row = ipiv[i] - 1; // 1-based to 0-based
    int current_row = i;

    if (pivot_row != current_row) {
      // Swap A[current_row, tid] with A[pivot_row, tid]
      // A is column-major, so A[row, col] is A + col * lda + row
      float *row1_ptr = A + tid * lda + current_row;
      float *row2_ptr = A + tid * lda + pivot_row;

      float temp = *row1_ptr;
      *row1_ptr = *row2_ptr;
      *row2_ptr = temp;
    }
  }
}

// ========================================================================================
// HELPER FUNCTIONS
// ========================================================================================
void cast_fp32_to_fp16_strided(const float *d_src, int lda, __half *d_dst,
                               int rows, int cols) {
  dim3 block(32, 32);
  // Grid X -> Rows, Grid Y -> Cols
  dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
  cast_fp32_to_fp16_strided_kernel<<<grid, block>>>(d_src, lda, d_dst, rows,
                                                    cols);
}

/**
 * Apply Pivots (LASWP equivalent)
 * Swaps rows in the trailing submatrix based on ipiv buffer.
 * A_panel_start points to A[k, k+B] (The top-left of the trailing matrix
 * 'Right') BUT we need to swap rows starting from ROW k. So we pass the pointer
 * to the requested column start, but row 0? No, simpler: Pass pointer to A[0,
 * k+B].
 *
 * wait. Standard LAPACK apply_pivots applies to the whole row?
 * We only need to swap rows in the columns we haven't processed yet (k+B to N).
 * Columns 0 to k+B-1 are already factored/used. Do we need to swap them?
 * In standard LU (getrf), pivoting is applied to the WHOLE column usually?
 * Actually for "Right Looking", we only need to pivot the active trailing
 * matrix to preserve the submatrix property for the next step. However, the L
 * part (columns 0..k) should mathematically distinct.
 *
 * Let's stick to the Right-Looking update:
 * We swap rows in A[k:N, k+B:N].
 *
 * k: global offset of the current panel (row index offset)
 * ipiv: pivots relative to row k
 */
__global__ void apply_pivots_kernel(float *A, int lda, const int *ipiv,
                                    int num_pivots, int k, int cols_to_update) {
  int c = blockIdx.x * blockDim.x +
          threadIdx.x; // column index in the trailing matrix (0..rem-1)
  if (c >= cols_to_update)
    return;

  // The trailing matrix starts at A[0,0] offset properly by caller?
  // Let's assume A points to A[0, k+B].
  // We iterate through the 'num_pivots' pivots.
  for (int i = 0; i < num_pivots; ++i) {
    int pivot_idx_rel = ipiv[i] - 1; // 0-based relative to k
    int row1_abs = k + i;
    int row2_abs = k + pivot_idx_rel;

    if (row1_abs != row2_abs) {
      // Swap A[row1, c] with A[row2, c]
      float *p1 = A + c * lda + row1_abs;
      float *p2 = A + c * lda + row2_abs;
      float val = *p1;
      *p1 = *p2;
      *p2 = val;
    }
  }
}

void apply_pivots(float *d_A_base, int lda, const int *d_ipiv, int num_pivots,
                  int k, int cols_to_update) {
  if (cols_to_update <= 0)
    return;
  int blockSize = 256;
  int numBlocks = (cols_to_update + blockSize - 1) / blockSize;
  // d_A_base should point to the first column to be updated (col k+B), row 0.
  apply_pivots_kernel<<<numBlocks, blockSize>>>(d_A_base, lda, d_ipiv,
                                                num_pivots, k, cols_to_update);
}

__global__ void matrix_cast_fp64_to_fp32_kernel(const double *src, float *dst,
                                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * n) {
    dst[idx] = (float)src[idx];
  }
}

__global__ void matrix_cast_fp32_to_fp64_kernel(const float *src, double *dst,
                                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * n) {
    dst[idx] = (double)src[idx];
  }
}

// ========================================================================================
// MAIN SOLVER: FP16 TENSOR CORE ITERATIVE REFINEMENT
// ========================================================================================
int solve_tensor_core_ir(const double *A_host, const double *b_host,
                         double *x_host, int n, int max_iterations,
                         double tolerance, int *iterations_used,
                         MixedPrecisionTiming *timing, bool verbose) {

  // Initialize timing
  if (timing) {
    timing->factorization_ms = 0;
    timing->refinement_ms = 0;
    timing->total_ms = 0;
    timing->final_residual = 0;
  }

  cudaEvent_t start_total, stop_total, start_factor, stop_factor;
  cudaEventCreate(&start_total);
  cudaEventCreate(&stop_total);
  cudaEventCreate(&start_factor);
  cudaEventCreate(&stop_factor);

  cudaEventRecord(start_total);

  // Handles
  cusolverDnHandle_t solver_handle;
  cublasHandle_t blas_handle;
  cusolverDnCreate(&solver_handle);
  cublasCreate(&blas_handle);
  // CRITICAL: Enable Tensor Cores
  cublasSetMathMode(blas_handle, CUBLAS_TENSOR_OP_MATH);

  // 1. Allocate Memory
  double *d_A_fp64, *d_b_fp64, *d_x_fp64, *d_r_fp64, *d_d_fp64;
  float *d_A_fp32, *d_r_fp32;

  CUDA_CHECK(cudaMalloc(&d_A_fp64, n * n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_b_fp64, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_x_fp64, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_r_fp64, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_d_fp64, n * sizeof(double)));

  CUDA_CHECK(cudaMalloc(&d_A_fp32, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r_fp32, n * sizeof(float)));

  // Transpose Copy A and b to device (FP64)
  // OPTIMIZATION: Do transpose on GPU to avoid slow CPU loop (O(N^2))
  double *d_A_row;
  CUDA_CHECK(cudaMalloc(&d_A_row, n * n * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_A_row, A_host, n * n * sizeof(double),
                        cudaMemcpyHostToDevice));

  // Transpose d_A_row (Row Major) -> d_A_fp64 (Col Major)
  // Effectively A^T in Col Major is A in Row Major.
  // Use geam: C = alpha * op(A) + beta * op(B)
  double alpha_t = 1.0;
  double beta_t = 0.0;
  CUBLAS_CHECK(cublasDgeam(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n,
                           &alpha_t, d_A_row, n, &beta_t, d_A_row, n, d_A_fp64,
                           n));

  CUDA_CHECK(cudaFree(d_A_row));

  CUDA_CHECK(
      cudaMemcpy(d_b_fp64, b_host, n * sizeof(double), cudaMemcpyHostToDevice));

  // Initialize x = 0
  CUDA_CHECK(cudaMemset(d_x_fp64, 0, n * sizeof(double)));

  // ====================================================================================
  // 2. FACTORIZATION PHASE (Manual Block LU with Tall Panels)
  // ====================================================================================
  cudaEventRecord(start_factor);

  // 2.1 Convert FP64 -> FP32 (Full Matrix)
  {
    int blockSize = 256;
    int numBlocks = (n * n + blockSize - 1) / blockSize;
    matrix_cast_fp64_to_fp32_kernel<<<numBlocks, blockSize>>>(d_A_fp64,
                                                              d_A_fp32, n);
  }

  // 2.2 Block LU Loop
  int B = 2048; // Increase block size to reduce pivoting overhead

  // Buffers for FP16 update
  __half *d_L_panel_fp16, *d_U_panel_fp16;
  CUDA_CHECK(cudaMalloc(&d_L_panel_fp16, n * B * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_U_panel_fp16, B * n * sizeof(__half)));

  // Pre-allocate workspaces for diagonal factorization
  int lwork_block = 0;
  float *d_Diagonal_dummy = d_A_fp32;
  CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(
      solver_handle, n, B, d_Diagonal_dummy, n, &lwork_block));

  float *d_work_block;
  CUDA_CHECK(cudaMalloc(&d_work_block, lwork_block * sizeof(float)));
  int *d_ipiv_block;
  CUDA_CHECK(cudaMalloc(
      &d_ipiv_block, n * sizeof(int))); // Must hold min(M,N) for largest panel
  int *d_info_block;
  CUDA_CHECK(cudaMalloc(&d_info_block, sizeof(int)));

  float alpha_f = 1.0f;
  float minus_one_f = -1.0f;
  __half alpha_h = 1.0;
  __half minus_one_h = -1.0;
  __half beta_h = 0.0;

  for (int k = 0; k < n; k += B) {
    int actual_B = std::min(B, n - k);
    int rem = n - (k + actual_B); // Remaining columns/rows
    int rows_in_panel = n - k;

    // -------------------------------------------------------------
    // Step A: Factorize Tall Panel (FP32)
    // -------------------------------------------------------------
    // Panel starts at A[k, k]. Size: rows_in_panel x actual_B.
    float *d_Panel = d_A_fp32 + k * n + k;

    // Sgetrf on (N-k) x B matrix.
    CUSOLVER_CHECK(cusolverDnSgetrf(solver_handle, rows_in_panel, actual_B,
                                    d_Panel, n, d_work_block, d_ipiv_block,
                                    d_info_block));

    if (rem > 0) {
      // -------------------------------------------------------------
      // Step B: Apply Pivots to the Trailing Matrix (Right)
      // -------------------------------------------------------------
      // Offset: A + (k+B)*n (Start of first column of Right submatrix, row k
      // due to sgetrf pivot indexing?) Wait, cusolverDnSlaswp expects A to be
      // the start of the matrix. If pivots are relative to row k (because we
      // passed d_Panel which starts at row k), then we must pass d_Trailing
      // with row k pointer? 'd_ipiv_block' contains pivots 1-based. If index is
      // 'p', it swaps row 'p' with 'i'. If we passed d_Panel (row k), 'p' is
      // likely relative to k? YES. So we should pass pointer to A[k, k+B].

      float *d_Trailing_Rows_Start = d_A_fp32 + (k + actual_B) * n + k;
      CUSOLVER_CHECK(cusolverDnSlaswp(solver_handle, rem, d_Trailing_Rows_Start,
                                      n, 1, actual_B, d_ipiv_block, 1));

      // -------------------------------------------------------------
      // Step C & D: Update (TRSM + GEMM)
      // -------------------------------------------------------------
      // Step C: Update U_12 (Top-Right Block) -> TRSM
      // Solve L_11 * U_12 = A_12
      // L_11: B x B Unit Lower Triangular at A[k, k]
      // A_12: B x rem matrix at A[k, k+B] (After pivoting)

      float *d_L11 = d_A_fp32 + k * n + k;
      float *d_A12 = d_A_fp32 + (k + actual_B) * n + k;

      CUBLAS_CHECK(cublasStrsm(
          blas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
          CUBLAS_DIAG_UNIT, actual_B, rem, &alpha_f, d_L11, n, d_A12, n));

      // Step D: Schur Complement Update (FP16 GEMM)
      // A_22 -= L_21 * U_12
      // A_22: (N-k-B) x rem at A[k+B, k+B]
      // L_21: (N-k-B) x B at A[k+B, k]
      // U_12: B x rem at A[k, k+B] (Which was just updated by TRSM)

      int m_update = n - (k + actual_B); // rows in A22
      int n_update = rem;                // cols in A22
      int k_update = actual_B;           // inner dim

      float *d_A22 = d_A_fp32 + (k + actual_B) * n + (k + actual_B);
      float *d_L21 = d_A_fp32 + k * n + (k + actual_B);
      float *d_U12 = d_A_fp32 + (k + actual_B) * n + k;

      // Cast L_21 to FP16
      cast_fp32_to_fp16_strided(d_L21, n, d_L_panel_fp16, m_update, k_update);

      // Cast U_12 to FP16
      cast_fp32_to_fp16_strided(d_U12, n, d_U_panel_fp16, k_update, n_update);

      // Force Tensor Core Usage (Re-applying fix)
      cublasSetMathMode(blas_handle, CUBLAS_TENSOR_OP_MATH);

      // GEMM
      CUBLAS_CHECK(cublasGemmEx(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_update,
                                n_update, k_update, &minus_one_f,
                                d_L_panel_fp16, CUDA_R_16F, m_update, // A (L21)
                                d_U_panel_fp16, CUDA_R_16F, k_update, // B (U12)
                                &alpha_f, d_A22, CUDA_R_32F, n,       // C (A22)
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
  }

  cudaEventRecord(stop_factor);
  cudaEventSynchronize(stop_factor);
  float factor_ms = 0;
  cudaEventElapsedTime(&factor_ms, start_factor, stop_factor);
  if (timing)
    timing->factorization_ms = factor_ms;

  // cleanup temp implementation buffers
  cudaFree(d_L_panel_fp16);
  cudaFree(d_U_panel_fp16);
  cudaFree(d_work_block);
  cudaFree(d_ipiv_block);
  cudaFree(d_info_block);

  // ====================================================================================
  // 3. INITIAL SOLVE (FP32)
  // ====================================================================================
  // d_A_fp32 now contains L and U factors (approximate).
  // Solve Ax = b -> LUx = b

  // Copy b_fp64 -> x_fp32 (as RHS workspace)
  float *d_x_fp32;
  CUDA_CHECK(cudaMalloc(&d_x_fp32, n * sizeof(float)));

  {
    int blockSize = 512;
    int numBlocks = (n + blockSize - 1) / blockSize;
    matrix_cast_fp64_to_fp32_kernel<<<numBlocks, blockSize>>>(d_b_fp64,
                                                              d_x_fp32, n);
  }

  // Forward Solve: L * y = b. L is Unit Lower.
  CUBLAS_CHECK(cublasStrsm(
      blas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
      CUBLAS_DIAG_UNIT, n, 1, &alpha_f, d_A_fp32, n, d_x_fp32, n));

  // Backward Solve: U * x = y. U is Non-Unit Upper.
  CUBLAS_CHECK(cublasStrsm(
      blas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
      CUBLAS_DIAG_NON_UNIT, n, 1, &alpha_f, d_A_fp32, n, d_x_fp32, n));

  // Cast result to FP64
  {
    int blockSize = 512;
    int numBlocks = (n + blockSize - 1) / blockSize;
    matrix_cast_fp32_to_fp64_kernel<<<numBlocks, blockSize>>>(d_x_fp32,
                                                              d_x_fp64, n);
  }
  cudaFree(d_x_fp32);

  // ====================================================================================
  // 4. ITERATIVE REFINEMENT LOOP (FP64)
  // ====================================================================================
  double beta_zero = 0.0;
  cublasDscal(blas_handle, n, &beta_zero, d_r_fp64, 1);

  double nrm_b = 0.0;
  cublasDnrm2(blas_handle, n, d_b_fp64, 1, &nrm_b);

  int k = 0;
  for (k = 0; k < max_iterations; k++) {
    // 4.1 Compute r = b - Ax (FP64)
    // r = b
    CUDA_CHECK(cudaMemcpy(d_r_fp64, d_b_fp64, n * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    // r = r - A * x
    double alpha_d = -1.0;
    double beta_d = 1.0;

    // d_A_fp64 is the original matrix. d_x_fp64 is current solution.
    CUBLAS_CHECK(cublasDgemv(blas_handle, CUBLAS_OP_N, n, n, &alpha_d, d_A_fp64,
                             n, d_x_fp64, 1, &beta_d, d_r_fp64, 1));

    // Check convergence
    double nrm_r = 0.0;
    CUBLAS_CHECK(cublasDnrm2(blas_handle, n, d_r_fp64, 1, &nrm_r));
    if (timing)
      timing->final_residual = nrm_r / nrm_b;

    if (verbose) {
      std::cout << "  [Iter " << k << "] Residual: " << std::scientific
                << std::setprecision(2) << (nrm_r / nrm_b)
                << ((nrm_r / nrm_b < tolerance) ? " (Converged)" : "")
                << std::endl;
    }

    if (nrm_r / nrm_b < tolerance) {
      break;
    }

    // 4.2 Solve correction Ad = r (approximated uses FP32 LU)
    // Cast r (FP64) -> r (FP32)
    {
      int blockSize = 512;
      int numBlocks = (n + blockSize - 1) / blockSize;
      matrix_cast_fp64_to_fp32_kernel<<<numBlocks, blockSize>>>(d_r_fp64,
                                                                d_r_fp32, n);
    }

    // Manual TRSM Solve again
    // Forward
    CUBLAS_CHECK(cublasStrsm(
        blas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
        CUBLAS_DIAG_UNIT, n, 1, &alpha_f, d_A_fp32, n, d_r_fp32, n));
    // Backward
    CUBLAS_CHECK(cublasStrsm(
        blas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, n, 1, &alpha_f, d_A_fp32, n, d_r_fp32, n));

    // Cast d (FP32) -> d (FP64)
    // d_r_fp32 holds the result 'd' in single precision.
    {
      int blockSize = 512;
      int numBlocks = (n + blockSize - 1) / blockSize;
      matrix_cast_fp32_to_fp64_kernel<<<numBlocks, blockSize>>>(d_r_fp32,
                                                                d_d_fp64, n);
    }

    // 4.3 Update x = x + d (FP64)
    double alpha_one = 1.0;
    CUBLAS_CHECK(
        cublasDaxpy(blas_handle, n, &alpha_one, d_d_fp64, 1, d_x_fp64, 1));
  }

  if (iterations_used)
    *iterations_used = k;

  // Copy result back
  CUDA_CHECK(
      cudaMemcpy(x_host, d_x_fp64, n * sizeof(double), cudaMemcpyDeviceToHost));

  cudaEventRecord(stop_total);
  cudaEventSynchronize(stop_total);

  float total_ms = 0;
  cudaEventElapsedTime(&total_ms, start_total, stop_total);

  if (timing) {
    timing->total_ms = total_ms;
    timing->refinement_ms =
        total_ms - timing->factorization_ms; // Rough estimate
  }

  // Cleanup
  cudaFree(d_A_fp64);
  cudaFree(d_b_fp64);
  cudaFree(d_x_fp64);
  cudaFree(d_r_fp64);
  cudaFree(d_d_fp64);
  cudaFree(d_A_fp32);
  cudaFree(d_r_fp32);

  cusolverDnDestroy(solver_handle); // correct casing
  cublasDestroy(blas_handle);
  return 0;
}
