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
    MPI_Offset offset = mpiData.displacements[mpiData.rank];
    MPI_File fh;
    MPI_Status status;

    MPI_File_open(MPI_COMM_WORLD, filepath.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    std::vector<value_t> data(mpiData.lengths[mpiData.rank]);
    MPI_File_read_at(fh, offset, data.data(), mpiData.lengths[mpiData.rank], mpi_type_t, &status);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&fh);

    if (data.size() == 0)
    {
        if (mpiData.rank == 0)
            throw std::runtime_error("Zero data points were read in from the given file.");

        exit(1);
    }

    return data;
}