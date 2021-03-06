cmake_minimum_required(VERSION 3.0)

project(HPKmeans VERSION 1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS "-Wall -Wpedantic -fopenmp")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS "-std=gnu++1z -Wall -Wpedantic -fopenmp -L/usr/lib64 -lstdc++ -lm")
endif()

include_directories(include)

add_subdirectory(libs)
add_subdirectory(src)

add_executable(kmeans_serial_lloyd src/main.cpp)
add_executable(kmeans_omp_lloyd src/main.cpp)
add_executable(kmeans_mpi_lloyd src/main.cpp)
add_executable(kmeans_hybrid_lloyd src/main.cpp)
add_executable(kmeans_serial_optlloyd src/main.cpp)
add_executable(kmeans_omp_optlloyd src/main.cpp)
add_executable(kmeans_mpi_optlloyd src/main.cpp)
add_executable(kmeans_hybrid_optlloyd src/main.cpp)
add_executable(kmeans_serial_coreset_optlloyd src/main.cpp)
add_executable(kmeans_omp_coreset_optlloyd src/main.cpp)
add_executable(kmeans_mpi_coreset_optlloyd src/main.cpp)
add_executable(kmeans_hybrid_coreset_optlloyd src/main.cpp)

target_link_libraries(kmeans_serial_lloyd PUBLIC hpkmeans)
target_link_libraries(kmeans_omp_lloyd PUBLIC hpkmeans)
target_link_libraries(kmeans_mpi_lloyd PUBLIC hpkmeans)
target_link_libraries(kmeans_hybrid_lloyd PUBLIC hpkmeans)
target_link_libraries(kmeans_serial_optlloyd PUBLIC hpkmeans)
target_link_libraries(kmeans_omp_optlloyd PUBLIC hpkmeans)
target_link_libraries(kmeans_mpi_optlloyd PUBLIC hpkmeans)
target_link_libraries(kmeans_hybrid_optlloyd PUBLIC hpkmeans)
target_link_libraries(kmeans_serial_coreset_optlloyd PUBLIC hpkmeans)
target_link_libraries(kmeans_omp_coreset_optlloyd PUBLIC hpkmeans)
target_link_libraries(kmeans_mpi_coreset_optlloyd PUBLIC hpkmeans)
target_link_libraries(kmeans_hybrid_coreset_optlloyd PUBLIC hpkmeans)

target_compile_definitions(kmeans_serial_lloyd PUBLIC METHOD="lloyd" CORESET=0 PARALLELISM="serial")
target_compile_definitions(kmeans_omp_lloyd PUBLIC METHOD="lloyd" CORESET=0 PARALLELISM="omp")
target_compile_definitions(kmeans_mpi_lloyd PUBLIC METHOD="lloyd" CORESET=0 PARALLELISM="mpi")
target_compile_definitions(kmeans_hybrid_lloyd PUBLIC METHOD="lloyd" CORESET=0 PARALLELISM="hybrid")
target_compile_definitions(kmeans_serial_optlloyd PUBLIC METHOD="optlloyd" CORESET=0 PARALLELISM="serial")
target_compile_definitions(kmeans_omp_optlloyd PUBLIC METHOD="optlloyd" CORESET=0 PARALLELISM="omp")
target_compile_definitions(kmeans_mpi_optlloyd PUBLIC METHOD="optlloyd" CORESET=0 PARALLELISM="mpi")
target_compile_definitions(kmeans_hybrid_optlloyd PUBLIC METHOD="optlloyd" CORESET=0 PARALLELISM="hybrid")
target_compile_definitions(kmeans_serial_coreset_optlloyd PUBLIC METHOD="optlloyd" CORESET=1 PARALLELISM="serial")
target_compile_definitions(kmeans_omp_coreset_optlloyd PUBLIC METHOD="optlloyd" CORESET=1 PARALLELISM="omp")
target_compile_definitions(kmeans_mpi_coreset_optlloyd PUBLIC METHOD="optlloyd" CORESET=1 PARALLELISM="mpi")
target_compile_definitions(kmeans_hybrid_coreset_optlloyd PUBLIC METHOD="optlloyd" CORESET=1 PARALLELISM="hybrid")