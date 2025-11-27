#include "matrix_utils.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>

void generate_random_matrix_host(double* A, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n * n; i++) {
        A[i] = (double)rand() / RAND_MAX;
    }
    
    // Make diagonally dominant for better conditioning
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            if (i != j) row_sum += fabs(A[i * n + j]);
        }
        A[i * n + i] = row_sum + 1.0;  // Ensure diagonal dominance
    }
}

void generate_random_vector_host(double* x, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        x[i] = (double)rand() / RAND_MAX;
    }
}

void generate_identity_matrix_host(double* A, int n) {
    for (int i = 0; i < n * n; i++) {
        A[i] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        A[i * n + i] = 1.0;
    }
}

void generate_poisson_2d_host(double* A, int grid_n) {
    // Size of the system
    int n = grid_n * grid_n;
    
    // Initialize to zero
    for (int i = 0; i < n * n; i++) {
        A[i] = 0.0;
    }
    
    // Fill 5-point stencil
    // [ 4 -1  0 -1  0 ...]
    // [-1  4 -1  0 -1 ...]
    // ...
    for (int i = 0; i < grid_n; i++) {
        for (int j = 0; j < grid_n; j++) {
            int idx = i * grid_n + j;
            
            // Diagonal
            A[idx * n + idx] = 4.0;
            
            // Left neighbor
            if (j > 0) {
                int left_idx = i * grid_n + (j - 1);
                A[idx * n + left_idx] = -1.0;
            }
            
            // Right neighbor
            if (j < grid_n - 1) {
                int right_idx = i * grid_n + (j + 1);
                A[idx * n + right_idx] = -1.0;
            }
            
            // Top neighbor
            if (i > 0) {
                int top_idx = (i - 1) * grid_n + j;
                A[idx * n + top_idx] = -1.0;
            }
            
            // Bottom neighbor
            if (i < grid_n - 1) {
                int bottom_idx = (i + 1) * grid_n + j;
                A[idx * n + bottom_idx] = -1.0;
            }
        }
    }
}

void print_matrix(const double* A, int n, const char* name) {
    std::cout << name << " (" << n << "x" << n << "):" << std::endl;
    std::cout << std::scientific << std::setprecision(4);
    
    int max_print = std::min(n, 8);  // Only print up to 8x8
    for (int i = 0; i < max_print; i++) {
        for (int j = 0; j < max_print; j++) {
            std::cout << std::setw(12) << A[i * n + j] << " ";
        }
        if (max_print < n) std::cout << "...";
        std::cout << std::endl;
    }
    if (max_print < n) {
        std::cout << "..." << std::endl;
    }
    std::cout << std::endl;
}

void print_vector(const double* x, int n, const char* name) {
    std::cout << name << " (" << n << "):" << std::endl;
    std::cout << std::scientific << std::setprecision(6);
    
    int max_print = std::min(n, 10);
    std::cout << "[";
    for (int i = 0; i < max_print; i++) {
        std::cout << x[i];
        if (i < max_print - 1) std::cout << ", ";
    }
    if (max_print < n) std::cout << ", ...";
    std::cout << "]" << std::endl << std::endl;
}

double vector_norm_host(const double* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

double relative_error(const double* x, const double* x_ref, int n) {
    double* diff = new double[n];
    for (int i = 0; i < n; i++) {
        diff[i] = x[i] - x_ref[i];
    }
    
    double error_norm = vector_norm_host(diff, n);
    double ref_norm = vector_norm_host(x_ref, n);
    
    delete[] diff;
    
    return error_norm / ref_norm;
}

