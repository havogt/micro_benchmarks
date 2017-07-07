project(serialbox CXX CUDA)
cmake_minimum_required(VERSION 3.1)

set( CUDA_ARCH "sm_50" CACHE STRING "CUDA architecture" )

set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${CUDA_ARCH}" )

add_executable( main src/main.cu )

