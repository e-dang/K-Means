# cmake version to be used
cmake_minimum_required(VERSION 3.0)

# project name
project(HPKmeans VERSION 1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

# flags
SET(CMAKE_CXX_FLAGS "-Wall -Wpedantic -fopenmp")

find_package(MPI REQUIRED)
find_package(Boost 1.72.0 COMPONENTS random timer)

include_directories(include)

add_subdirectory(libs)
add_subdirectory(src)