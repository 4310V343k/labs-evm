add_library(integral_parallel_omp STATIC omp_parallel.cpp)

find_package(OpenMP REQUIRED)

target_include_directories(integral_parallel_omp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../common)

target_link_libraries(integral_parallel_omp PUBLIC common OpenMP::OpenMP_CXX)
