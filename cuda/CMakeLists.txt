cmake_minimum_required(VERSION 2.8)

project(cuda)
find_package(CUDA)
set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
    -O3
        -gencode arch=compute_62,code=sm_62
)
file(GLOB cudaSource *.cpp *.cu)
file(GLOB cudaInclude *.h)

cuda_add_library(cuda SHARED ${cudaSource})

add_subdirectory(chapter3)
add_subdirectory(chapter4)
