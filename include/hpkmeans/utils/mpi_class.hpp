#pragma once

#include <mpi.h>

namespace HPKmeans
{
template <typename precision, typename int_size>
class MPIClass
{
protected:
    MPI_Datatype mpi_precision;
    MPI_Datatype mpi_int_size;

public:
    MPIClass()
    {
        MPI_Type_match_size(MPI_TYPECLASS_REAL, sizeof(precision), &mpi_precision);
        MPI_Type_match_size(MPI_TYPECLASS_INTEGER, sizeof(int_size), &mpi_int_size);
    }

    virtual ~MPIClass() = default;
};
}  // namespace HPKmeans