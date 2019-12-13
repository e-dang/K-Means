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
include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")

# target
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
<<<<<<< HEAD
    add_executable(mpi_kmeans ./src/main_MPI.cpp ./src/Kmeans.cpp ./src/SerialKmeans.cpp ./src/OMPKmeans.cpp  ./src/MPIKmeans.cpp ./src/Reader.cpp ./src/Writer.cpp)
    add_executable(omp_kmeans ./src/main_omp.cpp ./src/SerialKMeans.cpp ./src/OMPKmeans.cpp ./src/Kmeans.cpp ./src/Reader.cpp ./src/Writer.cpp)
    target_link_libraries(mpi_kmeans ${Boost_LIBRARIES})
    target_link_libraries(omp_kmeans ${Boost_LIBRARIES})
=======
    add_executable(kmeans ./src/main_MPI.cpp ./src/Kmeans.cpp ./src/SerialKmeans.cpp ./src/OMPKmeans.cpp  ./src/MPIKmeans.cpp ./src/Reader.cpp ./src/Writer.cpp ./src/Coresets.cpp)
    target_link_libraries(kmeans ${Boost_LIBRARIES})
>>>>>>> f39c8b6a99e51748a69b6c8711954aebe9b833c2
endif()
