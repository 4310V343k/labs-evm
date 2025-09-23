cmake_minimum_required(VERSION 3.16)
project(weier_integral_mpi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI package
find_package(MPI REQUIRED)

# Add common library from 1_st_mt
add_subdirectory(../1_st_mt/cpp/common common)

# Create MPI integral library
add_library(integral_mpi STATIC mpi_parallel.cpp)

target_include_directories(integral_mpi PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ../1_st_mt/cpp/common
)

target_link_libraries(integral_mpi PUBLIC common MPI::MPI_CXX)

# Create MPI test application
add_executable(mpi_test mpi_test.cpp)
target_link_libraries(mpi_test integral_mpi common MPI::MPI_CXX)