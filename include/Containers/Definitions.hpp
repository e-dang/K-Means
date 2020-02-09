#pragma once

// typedef float value_t;

#define type_selection 1

#if type_selection == 1
    #define value_t float
    #define mpi_type_t MPI_FLOAT
#else
    #define value_t double
    #define mpi_type_t MPI_DOUBLE
#endif