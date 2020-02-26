# cmake version to be used
cmake_minimum_required(VERSION 3.0)

# project name
project(HPKmeans VERSION 1.0.0)


# flags
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -Wpedantic -fopenmp")


find_package(MPI REQUIRED)
find_package(Boost 1.72.0 COMPONENTS random)

add_subdirectory(./libs)

# include
include_directories(./include ./libs/include ${CMAKE_CURRENT_BINARY_DIR} ${MPI_INCLUDE_PATH} ${Boost_INCLUDE_DIRS})

# target
if(Boost_FOUND AND MPI_FOUND)
    add_executable(kmeans ./src/main.cpp
                          ./src/filesystem/file_rotator.cpp)
    target_link_libraries(kmeans sarge ${MPI_LIBRARIES} ${Boost_LIBRARIES})
endif()
