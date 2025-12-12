/**
 * Precision Benchmark
 * Measures raw throughput of different precision units on the GPU.
 */

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at "       \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUSOLVER_CHECK(call)                                                   \
  do {                                                                         \
    cusolverStatus_t status = call;                                            \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
      std::cerr << "cuSOLVER Error: " << status << " at " << __FILE__ << ":"   \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::cerr << "cuBLAS Error: " << status << " at " << __FILE__ << ":"     \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Helper to fill matrix with random data
template <typename T> void fill_random(T *data, int n) {
  for (int i = 0; i < n * n; i++) {
    data[i] = static_cast<T>((rand() % 100) / 100.0);
  }
}

void benchmark_fp64_lu(int n, cusolverDnHandle_t handle) {
  double *d_A;
  int *d_ipiv, *d_info;
  double *d_work;
  int lwork = 0;

  CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_ipiv, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

  CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(handle, n, n, d_A, n, &lwork));
  CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));

  // Warmup
  CUSOLVER_CHECK(
      cusolverDnDgetrf(handle, n, n, d_A, n, d_work, d_ipiv, d_info));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  CUSOLVER_CHECK(
      cusolverDnDgetrf(handle, n, n, d_A, n, d_work, d_ipiv, d_info));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  // FLOPs for LU is 2/3 * N^3
  double gflops = (2.0 / 3.0 * n * n * n) / (ms * 1e-3) / 1e9;

  std::cout << "FP64 (LU)   | N=" << n << " | Time: " << std::setw(7) << ms
            << " ms | TFLOPS: " << gflops / 1000.0 << std::endl;

  cudaFree(d_A);
  cudaFree(d_ipiv);
  cudaFree(d_info);
  cudaFree(d_work);
}

void benchmark_fp32_lu(int n, cusolverDnHandle_t handle) {
  float *d_A;
  int *d_ipiv, *d_info;
  float *d_work;
  int lwork = 0;

  CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ipiv, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

  CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(handle, n, n, d_A, n, &lwork));
  CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));

  // Warmup
  CUSOLVER_CHECK(
      cusolverDnSgetrf(handle, n, n, d_A, n, d_work, d_ipiv, d_info));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  CUSOLVER_CHECK(
      cusolverDnSgetrf(handle, n, n, d_A, n, d_work, d_ipiv, d_info));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  double gflops = (2.0 / 3.0 * n * n * n) / (ms * 1e-3) / 1e9;

  std::cout << "FP32 (LU)   | N=" << n << " | Time: " << std::setw(7) << ms
            << " ms | TFLOPS: " << gflops / 1000.0 << std::endl;

  cudaFree(d_A);
  cudaFree(d_ipiv);
  cudaFree(d_info);
  cudaFree(d_work);
}

void benchmark_fp16_gemm(int n, cublasHandle_t handle) {
  __half *d_A, *d_B, *d_C;
  // FP16 takes half the memory
  CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_C, n * n * sizeof(__half)));

  __half alpha = 1.0;
  __half beta = 0.0;

  // Warmup
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                            d_A, CUDA_R_16F, n, d_B, CUDA_R_16F, n, &beta, d_C,
                            CUDA_R_16F, n, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  // Standard GEMM: C = alpha * A * B + beta * C
  // This uses Tensor Cores on A100 (FP16 IN, FP32 Accumulate)
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                            d_A, CUDA_R_16F, n, d_B, CUDA_R_16F, n, &beta, d_C,
                            CUDA_R_16F, n, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  // GEMM is 2 * N^3
  double gflops = (2.0 * n * n * n) / (ms * 1e-3) / 1e9;

  std::cout << "FP16 (GEMM) | N=" << n << " | Time: " << std::setw(7) << ms
            << " ms | TFLOPS: " << gflops / 1000.0 << " (Note: GEMM, not LU)"
            << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

int main() {
  cusolverDnHandle_t solver_handle;
  cublasHandle_t blas_handle;

  cusolverDnCreate(&solver_handle);
  cublasCreate(&blas_handle);

  // Enable Tensor Cores for cuBLAS
  cublasSetMathMode(blas_handle, CUBLAS_TENSOR_OP_MATH);

  std::cout << "==========================================================="
            << std::endl;
  std::cout << "GPU PRECISION BENCHMARK (A100)" << std::endl;
  std::cout << "==========================================================="
            << std::endl;
  std::cout << "FP64/FP32 use LU Factorization (2/3 N^3 FLOPS)" << std::endl;
  std::cout << "FP16 uses Matrix Multiplication (2 N^3 FLOPS)" << std::endl;
  std::cout << "Goal: Measure Peak Throughput (TFLOPS)" << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;

  int sizes[] = {4096, 8192, 16384};

  for (int n : sizes) {
    benchmark_fp64_lu(n, solver_handle);
    benchmark_fp32_lu(n, solver_handle);
    benchmark_fp16_gemm(n, blas_handle);
    std::cout << "-----------------------------------------------------------"
              << std::endl;
  }

  cusolverDnDestroy(solver_handle);
  cublasDestroy(blas_handle);
  return 0;
}
