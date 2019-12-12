# cmake version to be used
cmake_minimum_required(VERSION 3.0)

# project name
project(Kmeans)

# Set the MPI compiler
set(CMAKE_CXX_COMPILER "mpic++")

# flags
SET(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wpedantic -fopenmp")

# include
include_directories(./include)
include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")

# target
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    add_executable(kmeans ./src/main.cpp ./src/Kmeans.cpp ./src/SerialKmeans.cpp ./src/OMPKmeans.cpp  ./src/MPIKmeans.cpp ./src/Reader.cpp ./src/Writer.cpp)
    target_link_libraries(kmeans ${Boost_LIBRARIES})
endif()
