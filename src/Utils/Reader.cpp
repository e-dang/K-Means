#include "Utils/Reader.hpp"

#include <fstream>
#include <iostream>

#include "mpi.h"

void VectorReader::read(std::string filepath, int_fast32_t numData, int_fast32_t numFeatures)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open specified file");
    }

    data = std::vector<value_t>(numData * numFeatures);
    file.read(reinterpret_cast<char*>(data.data()), sizeof(value_t) * numData * numFeatures);
}

void MPIReader::read(std::string filepath, int_fast32_t numData, int_fast32_t numFeatures)
{
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int_fast32_t numDataPerProc = numData * numFeatures / numProcs;
    MPI_Offset offset           = rank * numDataPerProc * sizeof(value_t);
    MPI_File fh;
    MPI_Status status;

    MPI_File_open(MPI_COMM_WORLD, filepath.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    data = std::vector<value_t>(numDataPerProc);
    MPI_File_read_at(fh, offset, data.data(), numDataPerProc, MPI_FLOAT, &status);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&fh);
}