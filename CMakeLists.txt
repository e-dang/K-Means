# cmake version to be used
cmake_minimum_required(VERSION 3.0)

# project name
project(Kmeans)

# flags
SET(CMAKE_CXX_FLAGS "-O3 -std=c++11 -Wall -Wpedantic -fopenmp")

# include
include_directories(./include)

# target
add_executable(kmeans ./src/Kmeans.cpp)