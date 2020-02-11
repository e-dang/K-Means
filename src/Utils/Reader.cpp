#include "Utils/Reader.hpp"

#include <fstream>
#include <iostream>

#include "Utils/Utils.hpp"
#include "mpi.h"

std::vector<value_t> VectorReader::read(const std::string& filepath, const int32_t& numData, const int32_t& numFeatures)
{
    std::ifstream file(filepath, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open specified file");
        exit(1);
    }

    std::vector<value_t> data(numData * numFeatures);
    file.read(reinterpret_cast<char*>(data.data()), sizeof(value_t) * numData * numFeatures);
    file.close();
    return data;
}

std::vector<value_t> MPIReader::read(const std::string& filepath, const int32_t& numData, const int32_t& numFeatures)
{
    auto mpiData      = getMPIData(numData);
    auto lengths      = mpiData.lengths[mpiData.rank] * numFeatures;
    MPI_Offset offset = mpiData.displacements[mpiData.rank] * numFeatures * sizeof(value_t);
    MPI_File fh;
    MPI_Status status;

    MPI_File_open(MPI_COMM_WORLD, filepath.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    std::vector<value_t> data(lengths);
    MPI_File_read_at(fh, offset, data.data(), lengths, mpi_type_t, &status);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&fh);

    int32_t count;
    MPI_Get_count(&status, MPI_INT, &count);
    MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (numData * numFeatures != count / (sizeof(value_t) / sizeof(float)) || count < 0)
    {
        if (mpiData.rank == 0)
            std::cerr << count / mpiData.numProcs << " data points were read in from the given file, but " << numData
                      << " data points were specified." << std::endl;
        exit(1);
    }

    return data;
}