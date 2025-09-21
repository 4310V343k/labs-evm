add_library(integral_single STATIC single.cpp)

target_include_directories(integral_single PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../common)

target_link_libraries(integral_single PUBLIC common)
