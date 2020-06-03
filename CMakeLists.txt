# cmake version to be used
cmake_minimum_required(VERSION 3.0)

# project name
project(HPKmeans VERSION 1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

# flags
SET(CMAKE_CXX_FLAGS "-O3 -Wall -Wpedantic -fopenmp")

find_package(MPI REQUIRED)
find_package(Boost 1.72.0 COMPONENTS random timer)

add_subdirectory(./libs)

# include
include_directories(./include ./libs/include ${CMAKE_CURRENT_BINARY_DIR} ${MPI_INCLUDE_PATH} ${Boost_INCLUDE_DIRS})

# target
if(Boost_FOUND AND MPI_FOUND)
    # add_executable(kmeans ./src/main.cpp
    #                       ./src/filesystem/file_rotator.cpp)
    add_executable(new_kmeans ./src/static_kmeans.cpp ./src/types/parallelism.cpp ./src/utils/file_rotator.cpp)
    # target_link_libraries(kmeans sarge ${MPI_LIBRARIES} ${Boost_LIBRARIES})
    target_link_libraries(new_kmeans matrix ${MPI_LIBRARIES} ${Boost_LIBRARIES})
endif()
