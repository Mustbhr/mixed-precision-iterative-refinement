#include "solver.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <cublas_v2.h>
#include <cusolverDn.h>

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return -1; \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error: " << status \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return -1; \
        } \
    } while(0)

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "cuSOLVER Error: " << status \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return -1; \
        } \
    } while(0)

int solve_lu_fp64(
    const double* A_host,
    const double* b_host,
    double* x_host,
    int n,
    double* time_ms
) {
    cudaEvent_t start, stop;
    if (time_ms) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }
    
    // Create cuSOLVER handle
    cusolverDnHandle_t solver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&solver_handle));
    
    // Allocate device memory
    double *d_A, *d_b;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    
    // Copy A and b to device (cuSOLVER uses column-major!)
    // Need to transpose A from row-major to column-major
    double* A_col_major = new double[n * n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_col_major[j * n + i] = A_host[i * n + j];  // Transpose
        }
    }
    
    CUDA_CHECK(cudaMemcpy(d_A, A_col_major, n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b_host, n * sizeof(double), cudaMemcpyHostToDevice));
    delete[] A_col_major;
    
    // Allocate workspace
    int* d_ipiv;  // Pivot indices
    int* d_info;  // Info
    CUDA_CHECK(cudaMalloc(&d_ipiv, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    // Query workspace size for LU factorization
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(solver_handle, n, n, d_A, n, &lwork));
    
    double* d_work;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));
    
    // LU factorization: A = P*L*U
    CUSOLVER_CHECK(cusolverDnDgetrf(solver_handle, n, n, d_A, n, d_work, d_ipiv, d_info));
    
    // Check if factorization was successful
    int info_h;
    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_h != 0) {
        std::cerr << "LU factorization failed, info = " << info_h << std::endl;
        cudaFree(d_A); cudaFree(d_b); cudaFree(d_ipiv); cudaFree(d_info); cudaFree(d_work);
        cusolverDnDestroy(solver_handle);
        return -1;
    }
    
    // Solve Ax = b using the LU factors
    CUSOLVER_CHECK(cusolverDnDgetrs(solver_handle, CUBLAS_OP_N, n, 1, 
                                    d_A, n, d_ipiv, d_b, n, d_info));
    
    // Copy solution back (it's in d_b)
    CUDA_CHECK(cudaMemcpy(x_host, d_b, n * sizeof(double), cudaMemcpyDeviceToHost));
    
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
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_ipiv);
    cudaFree(d_info);
    cudaFree(d_work);
    cusolverDnDestroy(solver_handle);
    
    return 0;
}

double compute_residual(
    const double* A_host,
    const double* x_host,
    const double* b_host,
    int n,
    double* r_host
) {
    // Allocate residual if not provided
    bool allocated = false;
    if (!r_host) {
        r_host = new double[n];
        allocated = true;
    }
    
    // Compute r = b - Ax
    // r = b
    memcpy(r_host, b_host, n * sizeof(double));
    
    // r = r - Ax (using simple matrix-vector multiply on CPU)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            r_host[i] -= A_host[i * n + j] * x_host[j];
        }
    }
    
    // Compute norm
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        norm += r_host[i] * r_host[i];
    }
    norm = sqrt(norm);
    
    if (allocated) {
        delete[] r_host;
    }
    
    return norm;
}

