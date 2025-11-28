cmake_minimum_required(VERSION 3.18)
project(MixedPrecisionIR CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
# Support wide range of GPUs: K80 (3.7), P100 (6.0), V100 (7.0), A100 (8.0), RTX 4090 (8.9)
# Note: Tensor Cores only available on 7.0+ (Volta/Turing/Ampere/Ada/Hopper)
set(CMAKE_CUDA_ARCHITECTURES 35 37 60 61 70 75 80 86 89)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Add subdirectories
add_subdirectory(src)

# Optional: Enable testing
enable_testing()

