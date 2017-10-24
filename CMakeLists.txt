project(serialbox CXX CUDA)
cmake_minimum_required(VERSION 3.1)

set( CUDA_ARCH "sm_50" CACHE STRING "CUDA architecture" )

set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${CUDA_ARCH}" )

add_executable( main src/main.cu )

include_directories( . )
add_executable( test_pointer_chaser test/test_pointer_chaser.cu )

add_executable( dev_sync test/dev_sync/dev_sync.cu )
set_property(TARGET dev_sync PROPERTY CUDA_SEPARABLE_COMPILATION ON)

