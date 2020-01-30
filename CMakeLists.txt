# cmake version to be used
cmake_minimum_required(VERSION 3.0)

# project name
project(Kmeans)

# Set the MPI compiler
set(CMAKE_CXX_COMPILER "mpic++")

# flags
SET(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wpedantic -fopenmp")
find_package(Boost 1.71.0 COMPONENTS random math)

# include
include_directories(./include)

# target
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    add_executable(kmeans ./src/main.cpp ./src/Kmeans.cpp ./src/Lloyd.cpp ./src/KPlusPlus.cpp ./src/Reader.cpp ./src/Writer.cpp ./include/DistanceFunctors.hpp ./src/KmeansAlgorithms.cpp ./src/ClusteringUpdater.cpp ./src/ClosestClusterFinder.cpp ./src/RandomSelector.cpp ./src/Averager.cpp)
    target_link_libraries(kmeans ${Boost_LIBRARIES})
endif()
