cmake_minimum_required(VERSION 3.10)
project(deformable_attention CUDA CXX)

# Find packages
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define include directories
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# CUDA settings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86 -O2")

# Define source files
set(SOURCES
    deform_attn.cpp
    deform_attn_kernel.cu
)

# Create CUDA library
add_library(deform_attn_cuda SHARED ${SOURCES})
target_link_libraries(deform_attn_cuda ${TORCH_LIBRARIES})

# Install library
install(TARGETS deform_attn_cuda DESTINATION lib)
install(FILES deformable_attention_cuda.py DESTINATION .) 