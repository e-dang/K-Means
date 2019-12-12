# cmake version to be used
cmake_minimum_required(VERSION 2.8)

# project name
project(Kmeans)

# Set the MPI compiler
set(CMAKE_CXX_COMPILER "mpic++")

# flags
SET(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")
find_package(Boost 1.41.0 COMPONENTS random math)

# include
include_directories(./include)
include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")

# target
include_directories(${Boost_INCLUDE_DIRS})
add_executable(kmeans ./src/main_MPI.cpp ./src/SerialKmeans.cpp ./src/OMPKmeans.cpp  ./src/MPIKmeans.cpp ./src/Reader.cpp ./src/Writer.cpp ./src/Coresets.cpp ./src/Kmeans.cpp)
target_link_libraries(kmeans ${Boost_LIBRARIES})