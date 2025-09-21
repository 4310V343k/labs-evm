add_library(integral_parallel STATIC parallel.cpp)

target_include_directories(integral_parallel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../common)

target_link_libraries(integral_parallel PUBLIC common)
