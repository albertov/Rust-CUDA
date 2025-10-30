cmake_minimum_required(VERSION 3.18)
project(cooperative_groups_tests CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Target CUDA architectures: SM 6.0, 7.0, 8.0, 8.9
set(CMAKE_CUDA_ARCHITECTURES 60 70 80 89)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Common includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Test executables
add_executable(grid_sync_basic grid_sync_basic.cu)
add_executable(grid_sync_multiblock grid_sync_multiblock.cu)
add_executable(grid_sync_phases grid_sync_phases.cu)

# Configure CUDA properties for all targets
foreach(target grid_sync_basic grid_sync_multiblock grid_sync_phases)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )

    # Enable cooperative groups support
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --extended-lambda
            -rdc=true
            --generate-line-info
        >
    )

    # Link CUDA libraries
    target_link_libraries(${target} CUDA::cudart)
endforeach()

# Build message
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
