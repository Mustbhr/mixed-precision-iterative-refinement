# Matrix utilities (CPU code)
add_library(matrix_utils utils/matrix_utils.cpp)

# Baseline solver (CUDA code)
add_library(baseline_solver solvers/baseline_solver.cu)
target_link_libraries(baseline_solver 
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
)

# Main executable
add_executable(mixed_precision_ir main.cu)
target_link_libraries(mixed_precision_ir
    matrix_utils
    baseline_solver
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
)

# CUDA architectures are set in root CMakeLists.txt
# Inherits: 35,37,60,61,70,75,80,86,89

