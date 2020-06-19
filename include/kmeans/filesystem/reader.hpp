#pragma once

#include <mpi.h>

#include <fstream>
#include <iostream>
#include <kmeans/types/parallelism.hpp>
#include <kmeans/utils/utils.hpp>
#include <matrix/matrix.hpp>
#include <string>

namespace hpkmeans
{
template <typename T>
class IReader
{
public:
    virtual ~IReader() = default;

    virtual Matrix<T> read(const std::string& filepath, const int32_t& numData, const int32_t& numFeatures) = 0;
};

template <typename T, Parallelism Level>
class MatrixReader : public IReader<T>
{
public:
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

    Matrix<T> read(const std::string& filepath, const int32_t& numData, const int32_t& numFeatures) override
    {
        return readImpl(filepath, numData, numFeatures);
    }

private:
    template <Parallelism _Level = Level>
    std::enable_if_t<isSharedMemory(_Level), Matrix<T>> readImpl(const std::string& filepath, const int32_t& numData,
                                                                 const int32_t& numFeatures)
    {
        auto file = openFile(filepath);

        Matrix<T> data(numData, numFeatures, true);
        file.read(reinterpret_cast<char*>(data.data()), sizeof(T) * numData * numFeatures);
        file.close();

        return data;
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level), Matrix<T>> readImpl(const std::string& filepath, const int32_t& numData,
                                                                const int32_t& numFeatures)
    {
        int rank;
        int size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        auto chunk = numData / size;
        auto scrap = chunk + (numData % size);

        std::vector<int> lengths(size);
        for (int i = 0; i < size; ++i)
        {
            lengths[i] = chunk;
        }
        lengths[size - 1] = scrap;

        Matrix<T> data(lengths[rank], numFeatures, true);
        MPI_File file;
        MPI_File_open(MPI_COMM_WORLD, filepath.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
        MPI_File_read_ordered(file, data.data(), data.bytes(), MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&file);

        return data;
    }
};
}  // namespace hpkmeans