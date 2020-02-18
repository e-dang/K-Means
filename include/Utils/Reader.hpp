#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
#include "Utils/Utils.hpp"
#include "mpi.h"

namespace HPKmeans
{
template <typename precision = double, typename int_size = int32_t>
class IReader
{
public:
    virtual ~IReader() = default;

    virtual Matrix<precision, int_size> read(const std::string& filepath, const int_size& numData,
                                             const int_size& numFeatures) = 0;
};

template <typename precision = double, typename int_size = int32_t>
class MatrixReader : public IReader<precision, int_size>
{
public:
    ~MatrixReader() = default;

    Matrix<precision, int_size> read(const std::string& filepath, const int_size& numData,
                                     const int_size& numFeatures) override
    {
        auto file = openFile(filepath);

        Matrix<precision, int_size> data(numData, numFeatures);
        file.read(reinterpret_cast<char*>(data.data()), sizeof(precision) * numData * numFeatures);
        file.close();

        return data;
    }

    std::ifstream openFile(const std::string& filepath)
    {
        std::ifstream file(filepath, std::ios::in | std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "Unable to open file: " << filepath << std::endl;
            exit(1);
        }

        return file;
    }
};

template <typename precision = double, typename int_size = int32_t>
class MPIMatrixReader : public IReader<precision, int_size>
{
public:
    ~MPIMatrixReader() = default;

    Matrix<precision, int_size> read(const std::string& filepath, const int_size& numData,
                                     const int_size& numFeatures) override
    {
        auto mpiData      = getMPIData(numData);
        auto lengths      = mpiData.lengths[mpiData.rank] * numFeatures;
        MPI_Offset offset = mpiData.displacements[mpiData.rank] * numFeatures * sizeof(precision);
        MPI_File fh;
        MPI_Status status;

        MPI_File_open(MPI_COMM_WORLD, filepath.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

        Matrix<precision, int_size> data(mpiData.lengths[mpiData.rank], numFeatures);
        MPI_File_read_at(fh, offset, data.data(), lengths, mpi_type_t, &status);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_File_close(&fh);

        int_size count;
        MPI_Get_count(&status, MPI_INT, &count);
        MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (numData * numFeatures != count / (sizeof(precision) / sizeof(float)) || count < 0)
        {
            if (mpiData.rank == 0)
                std::cerr << count / mpiData.numProcs << " data points were read in from the given file, but "
                          << numData << " data points were specified." << std::endl;
            exit(1);
        }

        return data;
    }
};

}  // namespace HPKmeans