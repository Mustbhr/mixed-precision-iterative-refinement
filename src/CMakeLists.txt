# Source files
add_executable(mixed_precision_ir main.cu)

# Link libraries
target_link_libraries(mixed_precision_ir 
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
)

# Set CUDA architectures
set_target_properties(mixed_precision_ir PROPERTIES 
    CUDA_ARCHITECTURES "80;86;89"
)

