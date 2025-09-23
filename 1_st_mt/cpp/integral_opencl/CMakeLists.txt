add_library(integral_opencl STATIC opencl_impl.cpp)

target_include_directories(integral_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../common)

find_package(OpenCL REQUIRED)

target_link_libraries(integral_opencl PUBLIC common OpenCL::OpenCL)
