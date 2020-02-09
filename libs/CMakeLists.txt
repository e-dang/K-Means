add_library(sarge STATIC ./src/sarge/sarge.cpp)
target_include_directories(sarge PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/sarge/")
