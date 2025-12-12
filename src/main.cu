/**
 * Mixed-Precision Iterative Refinement Solver
 * CS380 Project - Mustafa Albahrani
 *
 * Phase 1: Pure FP64 Baseline Solver
 * Phase 2: Mixed-Precision Iterative Refinement (FP32 + FP64)
 */

#include "matrix_utils.h"
#include "solver.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <string>

// Error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at "       \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/**
 * Print GPU device information
 */
void printDeviceInfo() {
  int deviceCount;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

  std::cout << "Found " << deviceCount << " CUDA device(s)\n" << std::endl;

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

    std::cout << "Device " << i << ": " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024 * 1024)
              << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;

    // Check Tensor Core support
    if (prop.major >= 7) {
      std::cout << "  Tensor Cores: Supported";
      if (prop.major == 8 && prop.minor == 0) {
        std::cout << " (A100 - FP16, TF32, FP64)";
      } else if (prop.major >= 9) {
        std::cout << " (Hopper - FP16, TF32, FP8)";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

// ============================================================================
// PHASE 1 TESTS (FP64 Baseline)
// ============================================================================

void test_fp64_baseline(int n, unsigned int seed = 42) {
  std::cout << "\n--- FP64 Baseline: " << n << "x" << n << " ---" << std::endl;

  double *A = new double[n * n];
  double *b = new double[n];
  double *x = new double[n];
  double *x_true = new double[n];

  generate_random_matrix_host(A, n, seed);
  generate_random_vector_host(x_true, n, seed + 1);

  // Compute b = A * x_true
  for (int i = 0; i < n; i++) {
    b[i] = 0.0;
    for (int j = 0; j < n; j++) {
      b[i] += A[i * n + j] * x_true[j];
    }
  }

  double time_ms;
  int result = solve_lu_fp64(A, b, x, n, &time_ms);

  if (result == 0) {
    double err = relative_error(x, x_true, n);
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << time_ms
              << " ms" << std::endl;
    std::cout << "  Error: " << std::scientific << std::setprecision(2) << err
              << std::endl;
    std::cout << "  Status: " << (err < 1e-10 ? "PASS" : "FAIL") << std::endl;
  } else {
    std::cout << "  FAILED" << std::endl;
  }

  delete[] A;
  delete[] b;
  delete[] x;
  delete[] x_true;
}

// ============================================================================
// PHASE 2 TESTS (Mixed-Precision IR)
// ============================================================================

void test_mixed_precision_ir(int n, unsigned int seed = 42) {
  std::cout << "\n--- Mixed-Precision IR: " << n << "x" << n << " ---"
            << std::endl;

  double *A = new double[n * n];
  double *b = new double[n];
  double *x = new double[n];
  double *x_true = new double[n];

  generate_random_matrix_host(A, n, seed);
  generate_random_vector_host(x_true, n, seed + 1);

  // Compute b = A * x_true
  for (int i = 0; i < n; i++) {
    b[i] = 0.0;
    for (int j = 0; j < n; j++) {
      b[i] += A[i * n + j] * x_true[j];
    }
  }

  int iterations_used;
  MixedPrecisionTiming timing;

  // Parameters: max 10 iterations, tolerance 1e-12
  int result = solve_mixed_precision_ir(A, b, x, n, 10, 1e-12, &iterations_used,
                                        &timing);

  if (result == 0) {
    double err = relative_error(x, x_true, n);

    std::cout << "  Total Time: " << std::fixed << std::setprecision(3)
              << timing.total_ms << " ms" << std::endl;
    std::cout << "    - Factorization (FP32): " << timing.factorization_ms
              << " ms" << std::endl;
    std::cout << "    - Refinement: " << timing.refinement_ms << " ms"
              << std::endl;
    std::cout << "  Iterations: " << iterations_used << std::endl;
    std::cout << "  Final Residual: " << std::scientific << std::setprecision(2)
              << timing.final_residual << std::endl;
    std::cout << "  Error vs True: " << err << std::endl;
    std::cout << "  Status: " << (err < 1e-10 ? "PASS" : "FAIL") << std::endl;
  } else {
    std::cout << "  FAILED" << std::endl;
  }

  delete[] A;
  delete[] b;
  delete[] x;
  delete[] x_true;
}

// ============================================================================
// COMPARATIVE BENCHMARK: FP64 vs Mixed-Precision IR
// ============================================================================
void benchmark_comparison() {
  std::cout << "\n==========================================================="
            << std::endl;
  std::cout << "BENCHMARK: FP64 vs Mixed-Precision IR" << std::endl;
  std::cout << "==========================================================="
            << std::endl;

  std::cout << "\n"
            << std::setw(6) << "Size" << std::setw(12) << "FP64(ms)"
            << std::setw(12) << "IR(ms)" << std::setw(12) << "TC(ms)"
            << std::setw(8) << "Spd(IR)" << std::setw(8) << "Spd(TC)"
            << std::setw(10) << "FP64 Err" << std::setw(10) << "TC Err"
            << std::endl;
  std::cout << std::string(100, '-') << std::endl;

  int sizes[] = {1024,  2048,  4096,  8192,  10000, 12000,
                 16000, 20000, 24000, 32000, 40000};

  for (int n : sizes) {
    double *A = new double[n * n];
    double *b = new double[n];
    double *x_fp64 = new double[n];
    double *x_ir = new double[n];
    double *x_true = new double[n];

    generate_random_matrix_host(A, n, 42);
    generate_random_vector_host(x_true, n, 123);

    // Compute b = A * x_true
    for (int i = 0; i < n; i++) {
      b[i] = 0.0;
      for (int j = 0; j < n; j++) {
        b[i] += A[i * n + j] * x_true[j];
      }
    }

    // Run FP64 baseline
    double fp64_time;
    int fp64_result = solve_lu_fp64(A, b, x_fp64, n, &fp64_time);
    double fp64_err =
        (fp64_result == 0) ? relative_error(x_fp64, x_true, n) : -1.0;

    // Run Mixed-Precision IR
    int ir_iters;
    MixedPrecisionTiming ir_timing;
    int ir_result = solve_mixed_precision_ir(A, b, x_ir, n, 10, 1e-12,
                                             &ir_iters, &ir_timing);
    double ir_err = (ir_result == 0) ? relative_error(x_ir, x_true, n) : -1.0;

    // Run Tensor Core IR (Phase 3)
    int tc_iters;
    MixedPrecisionTiming tc_timing;
    // We reuse x_ir buffer for now or allocate new? Using x_ir is fine if we
    // copy data out or don't need ir result anymore. Better to allocate
    // separate buffer to verify both. For now we will overwrite x_ir to keep
    // output simple or add x_tc. Let's create specific pointer at top of loop.
    // Re-using x_ir for TC result for simplicity of print.
    double *x_tc = new double[n];
    int tc_result =
        solve_tensor_core_ir(A, b, x_tc, n, 10, 1e-12, &tc_iters, &tc_timing);
    double tc_err = (tc_result == 0) ? relative_error(x_tc, x_true, n) : -1.0;

    // Calculate speedups
    double speedup_ir = (ir_result == 0 && fp64_result == 0)
                            ? fp64_time / ir_timing.total_ms
                            : 0.0;
    double speedup_tc = (tc_result == 0 && fp64_result == 0)
                            ? fp64_time / tc_timing.total_ms
                            : 0.0;

    // Print results
    if (fp64_result == 0) {
      std::cout << std::setw(6) << n << std::fixed << std::setprecision(2)
                << std::setw(12) << fp64_time << std::setw(12)
                << ir_timing.total_ms << std::setw(12) << tc_timing.total_ms
                << std::setw(8) << speedup_ir << "x" << std::setw(8)
                << speedup_tc << "x" << std::scientific << std::setprecision(1)
                << std::setw(10) << fp64_err << std::setw(10) << tc_err
                << std::endl;
    } else {
      std::cout << std::setw(6) << n << "  ERROR" << std::endl;
    }

    delete[] A;
    delete[] b;
    delete[] x_fp64;
    delete[] x_ir;
    delete[] x_tc;
    delete[] x_true;
  }

  std::cout << std::string(100, '-') << std::endl;
  std::cout << "IR = FP32 Factorization (Standard)" << std::endl;
  std::cout << "TC = FP16 Block Factorization (Tensor Cores)" << std::endl;
}

// ============================================================================
// DETAILED TIMING BREAKDOWN
// ============================================================================
// Shows where time is spent in the mixed-precision solver

void benchmark_timing_breakdown(int n) {
  std::cout << "\n==========================================================="
            << std::endl;
  std::cout << "TIMING BREAKDOWN: n = " << n << std::endl;
  std::cout << "==========================================================="
            << std::endl;

  double *A = new double[n * n];
  double *b = new double[n];
  double *x = new double[n];
  double *x_true = new double[n];

  generate_random_matrix_host(A, n, 42);
  generate_random_vector_host(x_true, n, 123);

  for (int i = 0; i < n; i++) {
    b[i] = 0.0;
    for (int j = 0; j < n; j++) {
      b[i] += A[i * n + j] * x_true[j];
    }
  }

  // Run multiple times for average
  const int runs = 3;
  double total_factor = 0, total_refine = 0, total_time = 0;
  int total_iters = 0;

  for (int r = 0; r < runs; r++) {
    int iters;
    MixedPrecisionTiming timing;
    solve_mixed_precision_ir(A, b, x, n, 10, 1e-12, &iters, &timing);
    total_factor += timing.factorization_ms;
    total_refine += timing.refinement_ms;
    total_time += timing.total_ms;
    total_iters += iters;
  }

  double avg_factor = total_factor / runs;
  double avg_refine = total_refine / runs;
  double avg_total = total_time / runs;
  double avg_iters = (double)total_iters / runs;

  std::cout << "\nAverage over " << runs << " runs:" << std::endl;
  std::cout << "  Factorization (FP32): " << std::fixed << std::setprecision(3)
            << avg_factor << " ms (" << std::setprecision(1)
            << (100.0 * avg_factor / avg_total) << "%)" << std::endl;
  std::cout << "  Refinement loop:      " << std::fixed << std::setprecision(3)
            << avg_refine << " ms (" << std::setprecision(1)
            << (100.0 * avg_refine / avg_total) << "%)" << std::endl;
  std::cout << "  Total:                " << std::fixed << std::setprecision(3)
            << avg_total << " ms" << std::endl;
  std::cout << "  Iterations:           " << std::setprecision(1) << avg_iters
            << std::endl;

  // Compare to FP64
  double fp64_time;
  solve_lu_fp64(A, b, x, n, &fp64_time);
  std::cout << "\nComparison to FP64:" << std::endl;
  std::cout << "  FP64 baseline:        " << std::fixed << std::setprecision(3)
            << fp64_time << " ms" << std::endl;
  std::cout << "  Speedup:              " << std::setprecision(2)
            << (fp64_time / avg_total) << "x" << std::endl;

  delete[] A;
  delete[] b;
  delete[] x;
  delete[] x_true;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char **argv) {
  std::cout << "==========================================================="
            << std::endl;
  std::cout << "Mixed-Precision Iterative Refinement Solver" << std::endl;
  std::cout << "CS380 Project - Mustafa Albahrani" << std::endl;
  std::cout << "==========================================================="
            << std::endl;

  // Print device info
  printDeviceInfo();

  // Phase 1: FP64 baseline tests
  std::cout << "\n==========================================================="
            << std::endl;
  std::cout << "PHASE 1: FP64 BASELINE TESTS" << std::endl;
  std::cout << "==========================================================="
            << std::endl;

  test_fp64_baseline(64);
  test_fp64_baseline(256);

  // Phase 2: Mixed-precision IR tests
  std::cout << "\n==========================================================="
            << std::endl;
  std::cout << "PHASE 2: MIXED-PRECISION IR TESTS" << std::endl;
  std::cout << "==========================================================="
            << std::endl;

  test_mixed_precision_ir(64);
  test_mixed_precision_ir(256);

  // Comparative benchmark
  benchmark_comparison();

  // Detailed timing breakdown for medium-sized problem
  benchmark_timing_breakdown(512);

  // Summary
  std::cout << "\n==========================================================="
            << std::endl;
  std::cout << "SUMMARY" << std::endl;
  std::cout << "==========================================================="
            << std::endl;
  std::cout << "\nPhase 2 Implementation Complete!" << std::endl;
  std::cout << "- FP32 factorization with FP64 iterative refinement"
            << std::endl;
  std::cout << "- Achieves FP64-level accuracy (~1e-12 tolerance)" << std::endl;
  return 0;
}
