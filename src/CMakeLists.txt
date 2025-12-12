# Matrix utilities (CPU code)
add_library(matrix_utils utils/matrix_utils.cpp)

# Baseline solver - Pure FP64 (CUDA code)
add_library(baseline_solver solvers/baseline_solver.cu)
target_link_libraries(baseline_solver 
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
)

# Mixed-precision solver - FP32 factorization + FP64 refinement (CUDA code)
add_library(mixed_precision_solver solvers/mixed_precision_solver.cu)
target_link_libraries(mixed_precision_solver 
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
)

# Main executable
add_executable(mixed_precision_ir main.cu)
target_link_libraries(mixed_precision_ir
    matrix_utils
    baseline_solver
    mixed_precision_solver
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
)

# Precision Benchmark
add_executable(precision_benchmark precision_benchmark.cu)
target_link_libraries(precision_benchmark
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
)

# CUDA architectures are set in root CMakeLists.txt
# Inherits: 35,37,60,61,70,75,80,86,89

