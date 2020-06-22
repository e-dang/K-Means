find_package(MPI REQUIRED)
find_package(Boost 1.71.0 COMPONENTS random timer)

if(Boost_FOUND AND MPI_FOUND)
    add_executable(kmeans main.cpp
                          types/parallelism.cpp
                          filesystem/file_rotator.cpp)
    target_link_libraries(kmeans matrix ${MPI_LIBRARIES} ${Boost_LIBRARIES})
endif()