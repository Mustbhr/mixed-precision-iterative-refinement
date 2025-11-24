/**
 * Mixed-Precision Iterative Refinement Solver
 * CS380 Project - Mustafa Albahrani
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
            std::cout << "  ✓ Tensor Cores: Supported" << std::endl;
        } else {
            std::cout << "  ✗ Tensor Cores: Not Supported" << std::endl;
        }
        std::cout << std::endl;
    }
}

/**
 * Simple test: Matrix-vector multiplication
 * y = alpha * A * x + beta * y
 */
void testMatrixVectorMul() {
    std::cout << "Testing cuBLAS Matrix-Vector Multiplication..." << std::endl;
    
    const int n = 4;  // Small test size
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Host matrices
    float h_A[n*n] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float h_x[n] = {1, 2, 3, 4};
    float h_y[n] = {0, 0, 0, 0};
    
    // Device matrices
    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Perform y = A * x
    // Note: cuBLAS assumes column-major, but we're using row-major
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, n, n, 
                            &alpha, d_A, n, d_x, 1, &beta, d_y, 1));
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print result
    std::cout << "Result: [";
    for (int i = 0; i < n; i++) {
        std::cout << h_y[i];
        if (i < n-1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Expected: [30, 70, 110, 150]
    std::cout << "Expected: [30, 70, 110, 150]" << std::endl;
    
    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    
    std::cout << "✓ Test completed\n" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Mixed-Precision Iterative Refinement" << std::endl;
    std::cout << "CS380 GPU Programming Project" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Print device info
    printDeviceInfo();
    
    // Run a simple test
    testMatrixVectorMul();
    
    std::cout << "Next steps:" << std::endl;
    std::cout << "1. Implement matrix generation utilities" << std::endl;
    std::cout << "2. Implement LU factorization using cuSOLVER" << std::endl;
    std::cout << "3. Implement iterative refinement loop" << std::endl;
    std::cout << "4. Optimize with Tensor Cores" << std::endl;
    
    return 0;
}

