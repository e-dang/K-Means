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
                          ./src/Kmeans/Kmeans.cpp
                          ./src/Kmeans/KmeansFacade.cpp
                          ./src/KmeansAlgorithms/CoresetCreator.cpp
                          ./src/KmeansAlgorithms/KPlusPlus.cpp
                          ./src/KmeansAlgorithms/Lloyd.cpp
                          ./src/Strategies/Averager.cpp
                          ./src/Strategies/ClosestClusterFinder.cpp
                          ./src/Strategies/ClosestClusterUpdater.cpp
                          ./src/Strategies/ClusteringUpdater.cpp
                          ./src/Strategies/CoresetClusteringFinisher.cpp
                          ./src/Strategies/CoresetDistributionCalculator.cpp
                          ./src/Strategies/DistanceSumCalculator.cpp
                          ./src/Strategies/KmeansDataCreator.cpp
                          ./src/Strategies/PointReassigner.cpp
                          ./src/Strategies/RandomSelector.cpp
                          ./src/Factories/StrategyFactories.cpp
                          ./src/Factories/KmeansAlgorithmFactories.cpp
                          ./src/Factories/KmeansFactory.cpp
                          ./src/Utils/Reader.cpp
                          ./src/Utils/Writer.cpp
                          ./src/Utils/FileRotator.cpp
                          ./include/Utils/DistanceFunctors.hpp
                          ./include/Utils/Utils.hpp)
    target_link_libraries(kmeans sarge ${MPI_LIBRARIES} ${Boost_LIBRARIES})
endif()
