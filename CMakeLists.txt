cmake_minimum_required(VERSION 3.20)

# Removed because findPackage(FreeImage) only works on Linux
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")


if(DEFINED ENV{VCPKG_ROOT})
    message("VCPKG_ROOT is ${VCPKG_ROOT}")
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    message("VCPKG_ROOT $ENV{VCPKG_ROOT}")
    message("CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE}")
endif()

cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME ProjectName)
string(REPLACE " " "_" ProjectId ${ProjectName})

# Use current directory as project name
project(${ProjectName} VERSION 0.9 LANGUAGES CXX CUDA)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# FreeImage does not work with shared libraries!
set(BUILD_SHARED_LIBS FALSE)

# Position independent code
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G") # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries from CUDA sample source code
include_directories(
    Common
    Common/UtilNPP
)

# Find FreeImage
set(CMAKE_PREFIX_PATH /home/erwin/freeimage)
find_package(FreeImage CONFIG REQUIRED)

# Only create executable if FreeImage is found
if(${FreeImage_FOUND})

    message("FreeImage found.")
    # Add C++ and CUDA sources from src/
    file(GLOB source_files "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/include/*.cu")

    # Add C++ and CUDA header files from include/
    file(GLOB header_files "${CMAKE_SOURCE_DIR}/src/*.h" "${CMAKE_SOURCE_DIR}/include/*.cuh")

    # Add include directory "./include"
    include_directories(${CMAKE_SOURCE_DIR}/include)

    # Add target for boxFilterNPP
    add_executable(${PROJECT_NAME} ${source_files})

    # Add extended lambda to CUDA
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

    # Set standard to Cxx 17
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17 cuda_std_17)

    # Enable separable compilation
    #set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Include directories
    if(WIN32)
        message("Using vcpkg include directory ${VCPKG_IMPORT_PREFIX}/include")
        target_include_directories(${PROJECT_NAME} PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
            ${VCPKG_IMPORT_PREFIX}/include
        )
    else()
        message("Using default include directory")
        # Include directories + CUDA Toolkit + FreeImage
        target_include_directories(${PROJECT_NAME} PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
            ${VCPKG_IMPORT_PREFIX}/include
            ${FreeImage_INCLUDE_DIRS}
        )
    endif()

    if(WIN32 OR DEFINED (ENV{VCPKG_ROOT}))
        # Link libraries: CUDA (npp, nppisu, nppif, cudart) + FreeImage
        target_link_libraries(${PROJECT_NAME} PRIVATE
            CUDA::nppc
            CUDA::nppisu
            CUDA::nppif
            CUDA::nppig
            CUDA::cudart
            freeimage::FreeImage
        )
    else()
        # Link libraries: CUDA (npp, nppisu, nppif, cudart) + FreeImage
        target_link_libraries(${PROJECT_NAME} PRIVATE
            CUDA::nppc
            CUDA::nppisu
            CUDA::nppif
            CUDA::nppig
            CUDA::cudart
            freeimage::FreeImage
            ${FreeImage_LIBRARIES}
        )
    endif()

    message("Current binary dir ${CMAKE_CURRENT_BINARY_DIR}")
    message("Runtime output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

else()
    message(STATUS "FreeImage not found - will not build sample ${PROJECT_NAME}")
endif()
