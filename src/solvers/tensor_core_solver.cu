#include "solver.h"
#include <cmath>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cusolverDn.h>
#include <iostream>

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
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < rows && c < cols) {
    // Source is col-major: src[c * lda + r]
    // Dest is col-major contiguous: dst[c * rows + r]
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
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
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
                         MixedPrecisionTiming *timing) {

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
  double *A_col = new double[n * n];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      A_col[j * n + i] = A_host[i * n + j];

  CUDA_CHECK(cudaMemcpy(d_A_fp64, A_col, n * n * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_b_fp64, b_host, n * sizeof(double), cudaMemcpyHostToDevice));
  delete[] A_col;

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
    // L_11: B x B Unit Lower Triangular at A[k, k]
    // A_12: B x rem matrix at A[k, k+B] (After pivoting)
    // Note: A_12 is technically just the top B rows of the trailing columns.

    float *d_L11 = d_A_fp32 + k * n + k;
    float *d_A12 = d_A_fp32 + (k + actual_B) * n + k;

    CUBLAS_CHECK(cublasStrsm(
        blas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
        CUBLAS_DIAG_UNIT, actual_B, rem, &alpha_f, d_L11, n, d_A12, n));

    // -------------------------------------------------------------
    // Step D: Schur Complement Update (FP16 GEMM)
    // -------------------------------------------------------------
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
    // Dimensions: m_update x k_update
    cast_fp32_to_fp16_strided(d_L21, n, d_L_panel_fp16, m_update, k_update);

    // Cast U_12 to FP16
    // Dimensions: k_update x n_update
    cast_fp32_to_fp16_strided(d_U12, n, d_U_panel_fp16, k_update, n_update);

    // GEMM
    CUBLAS_CHECK(cublasGemmEx(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_update,
                              n_update, k_update, &minus_one_f, d_L_panel_fp16,
                              CUDA_R_16F, m_update,                 // A (L21)
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

// Copy b_fp64 -> x_fp64 (as RHS workspace) -> convert to x_fp32
// Or just allocate temp FP32 RHS
float *d_x_fp32;
CUDA_CHECK(cudaMalloc(&d_x_fp32, n * sizeof(float)));

// Cast b -> float
// reuse our strided kernel with rows=n, cols=1, lda=n?
// Just simple copy
{
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  matrix_cast_fp64_to_fp32_kernel<<<numBlocks, blockSize>>>(
      d_b_fp64, d_x_fp32, n); // treated as n x 1
}

// getrs needs ipiv.
// WAIT. We factorized in blocks. We did NOT produce a global ipiv.
// We produced `d_A_fp32` which has L and U in place, but WITHOUT global
// pivoting. So `cusolverDnSgetrs` WILL NOT WORK properly if we pass
// random/null ipiv. `cusolverDnSgetrs` expects the output of `Sgetrf`.

// SOLUTION: Since we manually computed LU (without pivoting), we must
// manually solve using TRSM. We cannot use `getrs`. Solve Ly = b (Forward)
// Solve Ux = y (Backward)

// Forward Solve: L * y = b. L is Unit Lower.
CUBLAS_CHECK(cublasStrsm(blas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                         CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, 1, &alpha_f,
                         d_A_fp32, n, d_x_fp32, n));

// Backward Solve: U * x = y. U is Non-Unit Upper.
CUBLAS_CHECK(cublasStrsm(blas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                         CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, 1, &alpha_f,
                         d_A_fp32, n, d_x_fp32, n));

// Cast result to FP64
{
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  matrix_cast_fp32_to_fp64_kernel<<<numBlocks, blockSize>>>(d_x_fp32, d_x_fp64,
                                                            n);
}
cudaFree(d_x_fp32); // done with initial solve

// ====================================================================================
// 4. ITERATIVE REFINEMENT LOOP (FP64)
// ====================================================================================
// Same logic as mixed_precision_solver.cu

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

  if (nrm_r / nrm_b < tolerance) {
    break;
  }

  // 4.2 Solve correction Ad = r (approximated uses FP32 LU)
  // Cast r (FP64) -> r (FP32)
  {
    int blockSize = 256;
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
    int blockSize = 256;
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
CUDA_CHECK(cudaMemcpy(x_host, d_x_fp64, n * sizeof(double),
                      cudaMemcpyDeviceToHost));

cudaEventRecord(stop_total);
cudaEventSynchronize(stop_total);
float total_ms = 0;
cudaEventElapsedTime(&total_ms, start_total, stop_total);

if (timing) {
  timing->total_ms = total_ms;
  timing->refinement_ms = total_ms - timing->factorization_ms; // Rough estimate
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
