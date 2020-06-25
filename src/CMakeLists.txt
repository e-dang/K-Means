find_package(MPI REQUIRED)
find_package(Boost 1.71.0 COMPONENTS random timer)


add_library(hpkmeans types/parallelism.cpp
                     filesystem/file_rotator.cpp)

target_link_libraries(hpkmeans PUBLIC matrix ${MPI_LIBRARIES} ${Boost_LIBRARIES})
