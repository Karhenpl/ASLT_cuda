cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CMAKE_CUDA_ARCHITECTURES 52)
    if(NOT CMAKE_CUDA_HOST_COMPILER)
        find_program(CUDA_COMPATIBLE_HOST_COMPILER NAMES g++-10 g++-9)
        if(CUDA_COMPATIBLE_HOST_COMPILER)
            set(CMAKE_CUDA_HOST_COMPILER "${CUDA_COMPATIBLE_HOST_COMPILER}")
        endif()
    endif()
    project(cuda_vector_sum LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cuda_vector_sum
    main.cpp
    vector_sum.cu
)

target_include_directories(cuda_vector_sum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})