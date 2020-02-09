#pragma once

#include <string>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"

class IReader
{
public:
    virtual ~IReader() {}

    virtual std::vector<value_t> read(const std::string& filepath, const int_fast32_t& numData,
                                      const int_fast32_t& numFeatures) = 0;
};

class VectorReader : public IReader
{
private:
    std::vector<value_t> data;

public:
    VectorReader() {}

    ~VectorReader() {}

    std::vector<value_t> read(const std::string& filepath, const int_fast32_t& numData,
                              const int_fast32_t& numFeatures) override;
};

class MPIReader : public IReader
{
private:
    std::vector<value_t> data;

public:
    MPIReader() {}

    ~MPIReader() {}

    std::vector<value_t> read(const std::string& filepath, const int_fast32_t& numData,
                              const int_fast32_t& numFeatures) override;
};