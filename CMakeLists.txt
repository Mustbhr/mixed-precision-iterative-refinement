cmake_minimum_required(VERSION 3.18)
project(MixedPrecisionIR CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
# Support V100 (7.0), RTX 6000 (7.5), A100 (8.0), and newer
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)

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

