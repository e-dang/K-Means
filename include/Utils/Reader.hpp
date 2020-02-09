#pragma once

#include <string>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"

class IReader
{
public:
    virtual ~IReader() {}

    virtual std::vector<value_t> read(const std::string& filepath, const int32_t& numData,
                                      const int32_t& numFeatures) = 0;
};

class VectorReader : public IReader
{
public:
    VectorReader() {}

    ~VectorReader() {}

    std::vector<value_t> read(const std::string& filepath, const int32_t& numData, const int32_t& numFeatures) override;
};

class MPIReader : public IReader
{
public:
    MPIReader() {}

    ~MPIReader() {}

    std::vector<value_t> read(const std::string& filepath, const int32_t& numData, const int32_t& numFeatures) override;
};