#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <cuda_runtime.h>

/**
 * Generate random dense matrix (host)
 * Uniform distribution [0, 1] with diagonal dominance
 */
void generate_random_matrix_host(double* A, int n, unsigned int seed = 42);

/**
 * Generate random vector (host)
 */
void generate_random_vector_host(double* x, int n, unsigned int seed = 123);

/**
 * Generate identity matrix (host)
 */
void generate_identity_matrix_host(double* A, int n);

/**
 * Generate 2D Poisson matrix (5-point stencil)
 * For -∇²u = f on grid_n×grid_n grid
 * Returns (grid_n²)×(grid_n²) matrix
 */
void generate_poisson_2d_host(double* A, int grid_n);

/**
 * Print matrix (small matrices only!)
 */
void print_matrix(const double* A, int n, const char* name = "Matrix");

/**
 * Print vector
 */
void print_vector(const double* x, int n, const char* name = "Vector");

/**
 * Compute ||x||₂ (L2 norm)
 */
double vector_norm_host(const double* x, int n);

/**
 * Compute relative error: ||x - x_ref|| / ||x_ref||
 */
double relative_error(const double* x, const double* x_ref, int n);

#endif // MATRIX_UTILS_H

