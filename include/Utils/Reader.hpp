#pragma once

#include <string>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"

class IReader
{
public:
    virtual ~IReader() {}

    virtual Matrix read(const std::string& filepath, const int32_t& numData, const int32_t& numFeatures) = 0;
};

class MatrixReader : public IReader
{
public:
    MatrixReader() {}

    ~MatrixReader() {}

    Matrix read(const std::string& filepath, const int32_t& numData, const int32_t& numFeatures) override;

    std::ifstream openFile(const std::string& filepath);
};

class MPIMatrixReader : public IReader
{
public:
    MPIMatrixReader() {}

    ~MPIMatrixReader() {}

    Matrix read(const std::string& filepath, const int32_t& numData, const int32_t& numFeatures) override;
};