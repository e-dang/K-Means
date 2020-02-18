# cmake version to be used
cmake_minimum_required(VERSION 3.0)

# project name
project(Kmeans)

# flags
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -Wpedantic -fopenmp")
add_subdirectory(./libs)
find_package(MPI REQUIRED)
find_package(Boost 1.72.0 COMPONENTS random)

# include
include_directories(./include ./libs/include ${MPI_INCLUDE_PATH} ${Boost_INCLUDE_DIRS})

# target
if(Boost_FOUND)
    add_executable(kmeans ./src/main.cpp
                          ./src/Utils/FileRotator.cpp)
    target_link_libraries(kmeans sarge ${MPI_LIBRARIES} ${Boost_LIBRARIES})
endif()
