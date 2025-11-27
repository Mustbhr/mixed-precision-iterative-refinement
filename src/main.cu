/**
 * Mixed-Precision Iterative Refinement Solver
 * CS380 Project - Mustafa Albahrani
 * 
 * Phase 1: Pure FP64 Baseline Solver
 */

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "matrix_utils.h"
#include "solver.h"

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error: " << status \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        
        // Check Tensor Core support (Compute Capability >= 7.0)
        if (prop.major >= 7) {
            std::cout << "  ✓ Tensor Cores: Supported";
            if (prop.major == 7 && prop.minor == 0) {
                std::cout << " (Volta - FP16)";
            } else if (prop.major == 7 && prop.minor == 5) {
                std::cout << " (Turing - FP16)";
            } else if (prop.major == 8 && prop.minor == 0) {
                std::cout << " (Ampere - FP16, TF32)";
            } else if (prop.major == 8 && prop.minor == 6) {
                std::cout << " (Ampere - FP16, TF32)";
            } else if (prop.major == 8 && prop.minor == 9) {
                std::cout << " (Ada Lovelace - FP16, TF32)";
            } else if (prop.major >= 9) {
                std::cout << " (Hopper - FP16, TF32, FP8)";
            }
            std::cout << std::endl;
        } else {
            std::cout << "  ✗ Tensor Cores: Not Supported (Pre-Volta)" << std::endl;
        }
        std::cout << std::endl;
    }
}

/**
 * Test 1: Identity matrix (simplest test)
 */
void test_identity_matrix() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test 1: Identity Matrix (4×4)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int n = 4;
    
    double* A = new double[n * n];
    double* b = new double[n];
    double* x = new double[n];
    double* x_true = new double[n];
    
    // Generate test: I*x = b, x = b
    generate_identity_matrix_host(A, n);
    for (int i = 0; i < n; i++) {
        b[i] = i + 1;      // b = [1, 2, 3, 4]
        x_true[i] = i + 1;  // Solution should be same
    }
    
    std::cout << "\nMatrix A (Identity):" << std::endl;
    print_matrix(A, n, "A");
    print_vector(b, n, "b");
    
    // Solve
    double time_ms;
    int result = solve_lu_fp64(A, b, x, n, &time_ms);
    
    if (result == 0) {
        print_vector(x, n, "x (computed)");
        
        std::cout << "Time: " << time_ms << " ms" << std::endl;
        
        // Check accuracy
        double err = relative_error(x, x_true, n);
        std::cout << "Relative error: " << std::scientific << std::setprecision(3) << err << std::endl;
        
        if (err < 1e-10) {
            std::cout << "✓ PASS: Accuracy excellent!" << std::endl;
        } else {
            std::cout << "✗ FAIL: Accuracy too low!" << std::endl;
        }
    } else {
        std::cout << "✗ Solver failed" << std::endl;
    }
    
    delete[] A; delete[] b; delete[] x; delete[] x_true;
}

/**
 * Test 2: Random matrix with known solution
 */
void test_random_matrix(int n) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test 2: Random Matrix (" << n << "×" << n << ")" << std::endl;
    std::cout << "========================================" << std::endl;
    
    double* A = new double[n * n];
    double* b = new double[n];
    double* x = new double[n];
    double* x_true = new double[n];
    
    // Generate test: A*x_true = b
    generate_random_matrix_host(A, n, 42);
    generate_random_vector_host(x_true, n, 123);
    
    // Compute b = A * x_true
    for (int i = 0; i < n; i++) {
        b[i] = 0.0;
        for (int j = 0; j < n; j++) {
            b[i] += A[i * n + j] * x_true[j];
        }
    }
    
    if (n <= 8) {
        std::cout << "\nMatrix A (Random, diagonally dominant):" << std::endl;
        print_matrix(A, n, "A");
    }
    
    // Solve
    double time_ms;
    int result = solve_lu_fp64(A, b, x, n, &time_ms);
    
    if (result == 0) {
        std::cout << "\n✓ Solver succeeded" << std::endl;
        std::cout << "Time: " << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
        
        // Check accuracy
        double err = relative_error(x, x_true, n);
        std::cout << "Relative error: " << std::scientific << std::setprecision(3) << err << std::endl;
        
        // Compute residual
        double res_norm = compute_residual(A, x, b, n);
        double b_norm = vector_norm_host(b, n);
        std::cout << "Residual norm: " << std::scientific << std::setprecision(3) << res_norm << std::endl;
        std::cout << "Relative residual: " << std::scientific << std::setprecision(3) << (res_norm / b_norm) << std::endl;
        
        if (err < 1e-10) {
            std::cout << "✓ PASS: Accuracy excellent!" << std::endl;
        } else if (err < 1e-6) {
            std::cout << "⚠ WARNING: Accuracy degraded but acceptable" << std::endl;
        } else {
            std::cout << "✗ FAIL: Accuracy too low!" << std::endl;
        }
    } else {
        std::cout << "✗ Solver failed" << std::endl;
    }
    
    delete[] A; delete[] b; delete[] x; delete[] x_true;
}

/**
 * Test 3: 2D Poisson matrix (structured, sparse-like)
 */
void test_poisson_matrix(int grid_n) {
    int n = grid_n * grid_n;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test 3: 2D Poisson Matrix (" << grid_n << "×" << grid_n << " grid = " << n << " DOF)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    double* A = new double[n * n];
    double* b = new double[n];
    double* x = new double[n];
    double* x_true = new double[n];
    
    // Generate Poisson matrix
    generate_poisson_2d_host(A, grid_n);
    generate_random_vector_host(x_true, n, 456);
    
    // Compute b = A * x_true
    for (int i = 0; i < n; i++) {
        b[i] = 0.0;
        for (int j = 0; j < n; j++) {
            b[i] += A[i * n + j] * x_true[j];
        }
    }
    
    if (n <= 64) {
        std::cout << "\nMatrix structure: 5-point stencil (-∇²u)" << std::endl;
    }
    
    // Solve
    double time_ms;
    int result = solve_lu_fp64(A, b, x, n, &time_ms);
    
    if (result == 0) {
        std::cout << "\n✓ Solver succeeded" << std::endl;
        std::cout << "Time: " << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
        
        // Check accuracy
        double err = relative_error(x, x_true, n);
        std::cout << "Relative error: " << std::scientific << std::setprecision(3) << err << std::endl;
        
        // Compute residual
        double res_norm = compute_residual(A, x, b, n);
        double b_norm = vector_norm_host(b, n);
        std::cout << "Relative residual: " << std::scientific << std::setprecision(3) << (res_norm / b_norm) << std::endl;
        
        if (err < 1e-10) {
            std::cout << "✓ PASS: Accuracy excellent!" << std::endl;
        } else if (err < 1e-6) {
            std::cout << "⚠ WARNING: Accuracy degraded but acceptable" << std::endl;
        } else {
            std::cout << "✗ FAIL: Accuracy too low!" << std::endl;
        }
    } else {
        std::cout << "✗ Solver failed" << std::endl;
    }
    
    delete[] A; delete[] b; delete[] x; delete[] x_true;
}

/**
 * Benchmark: Performance scaling
 */
void benchmark_sizes() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark: FP64 Performance Scaling" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n" << std::setw(8) << "Size" 
              << std::setw(12) << "Time(ms)" 
              << std::setw(12) << "GFLOPS"
              << std::setw(15) << "Rel. Error" << std::endl;
    std::cout << std::string(47, '-') << std::endl;
    
    int sizes[] = {64, 128, 256, 512, 1024};
    
    for (int n : sizes) {
        double* A = new double[n * n];
        double* b = new double[n];
        double* x = new double[n];
        double* x_true = new double[n];
        
        generate_random_matrix_host(A, n, 42);
        generate_random_vector_host(x_true, n, 123);
        
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
            // LU factorization complexity: 2/3 * n³ flops
            double flops = (2.0 / 3.0) * n * n * n;
            double gflops = (flops / 1e9) / (time_ms / 1000.0);
            double err = relative_error(x, x_true, n);
            
            std::cout << std::setw(8) << n 
                      << std::fixed << std::setprecision(3) << std::setw(12) << time_ms
                      << std::setw(12) << gflops
                      << std::scientific << std::setprecision(2) << std::setw(15) << err
                      << std::endl;
        } else {
            std::cout << std::setw(8) << n << "  FAILED" << std::endl;
        }
        
        delete[] A; delete[] b; delete[] x; delete[] x_true;
    }
    
    std::cout << std::endl;
    std::cout << "Note: GFLOPS calculated for LU factorization only (~2/3 n³ flops)" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Mixed-Precision Iterative Refinement" << std::endl;
    std::cout << "Phase 1: FP64 Baseline Solver" << std::endl;
    std::cout << "CS380 GPU Programming Project" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Print device info
    printDeviceInfo();
    
    // Run correctness tests
    test_identity_matrix();
    test_random_matrix(4);
    test_random_matrix(64);
    test_poisson_matrix(8);  // 8×8 grid = 64 DOF
    
    // Run performance benchmark
    benchmark_sizes();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "✓ Phase 1 Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nKey Results:" << std::endl;
    std::cout << "- FP64 baseline solver implemented" << std::endl;
    std::cout << "- Accuracy: ~1e-14 (double precision)" << std::endl;
    std::cout << "- Performance: see benchmark table above" << std::endl;
    std::cout << "\nNext Steps:" << std::endl;
    std::cout << "- Phase 2: Implement mixed-precision iterative refinement" << std::endl;
    std::cout << "- Goal: Match FP64 accuracy with 2-3x speedup using FP32" << std::endl;
    
    return 0;
}
