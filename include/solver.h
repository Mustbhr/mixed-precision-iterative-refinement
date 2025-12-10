#ifndef SOLVER_H
#define SOLVER_H

#include <cuda_runtime.h>
#include <cusolverDn.h>

/**
 * Timing breakdown for mixed-precision solver
 */
struct MixedPrecisionTiming {
  double factorization_ms; // Time for LU factorization (FP32)
  double refinement_ms;    // Time for all refinement iterations
  double total_ms;         // Total solve time
  int iterations;          // Number of refinement iterations used
  double final_residual;   // Final relative residual ||b-Ax||/||b||
};

/**
 * Pure FP64 solver using cuSOLVER LU factorization
 * Solves Ax = b using double precision
 *
 * Returns 0 on success, non-zero on failure
 */
int solve_lu_fp64(const double *A_host, // Input: n×n matrix (host, row-major)
                  const double *b_host, // Input: n×1 RHS vector (host)
                  double *x_host,       // Output: n×1 solution vector (host)
                  int n,                // Size of system
                  double *time_ms = nullptr // Optional: timing output
);

/**
 * Pure FP32 solver using cuSOLVER LU factorization
 * Input/output in FP64 for compatibility, but computation in FP32
 *
 * Returns 0 on success, non-zero on failure
 */
int solve_lu_fp32(const double *A_host, const double *b_host, double *x_host,
                  int n, double *time_ms = nullptr);

/**
 * Mixed-Precision Iterative Refinement solver
 *
 * Algorithm:
 *   1. Factor A in FP32 (fast)
 *   2. Solve initial x₀ in FP32
 *   3. Iteratively refine:
 *      - r = b - Ax (FP64)
 *      - Solve Ad = r (FP32, reuse factors)
 *      - x = x + d (FP64)
 *
 * Returns 0 on success, non-zero on failure
 */
int solve_mixed_precision_ir(
    const double *A_host, const double *b_host, double *x_host, int n,
    int max_iterations,          // Max refinement iterations (typically 10)
    double tolerance,            // Convergence tolerance (e.g., 1e-12)
    int *iterations_used,        // Output: actual iterations
    MixedPrecisionTiming *timing // Output: detailed timing breakdown
);

/**
 * Compute residual: r = b - Ax
 * Returns residual norm ||r||₂
 */
double compute_residual(const double *A_host, const double *x_host,
                        const double *b_host, int n,
                        double *r_host = nullptr // Optional: output residual
);

#endif // SOLVER_H
