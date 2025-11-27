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

# Set CUDA architectures (support V100, A100, etc.)
set_target_properties(mixed_precision_ir baseline_solver PROPERTIES 
    CUDA_ARCHITECTURES "70;75;80;86;89"
)

