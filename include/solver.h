#ifndef SOLVER_H
#define SOLVER_H

#include <cuda_runtime.h>
#include <cusolverDn.h>

/**
 * Pure FP64 solver using cuSOLVER LU factorization
 * Solves Ax = b using double precision
 * 
 * Returns 0 on success, non-zero on failure
 */
int solve_lu_fp64(
    const double* A_host,  // Input: n×n matrix (host, row-major)
    const double* b_host,  // Input: n×1 RHS vector (host)
    double* x_host,        // Output: n×1 solution vector (host)
    int n,                 // Size of system
    double* time_ms = nullptr  // Optional: timing output
);

/**
 * Compute residual: r = b - Ax
 * Returns residual norm ||r||₂
 */
double compute_residual(
    const double* A_host,
    const double* x_host,
    const double* b_host,
    int n,
    double* r_host = nullptr  // Optional: output residual vector
);

#endif // SOLVER_H

