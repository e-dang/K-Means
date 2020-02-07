#pragma once

#include <string>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"

class IReader
{
public:
    virtual void read(std::string filepath, int_fast32_t numData, int_fast32_t numFeatures) = 0;
};

class VectorReader : public IReader
{
private:
    std::vector<value_t> data;

public:
    VectorReader(){};
    ~VectorReader(){};

    void read(std::string filepath, int_fast32_t numData, int_fast32_t numFeatures) override;

    std::vector<value_t> getData() { return this->data; }
};

class MPIReader : public IReader
{
private:
    std::vector<value_t> data;

public:
    MPIReader(){};
    ~MPIReader(){};

    void read(std::string filepath, int_fast32_t numData, int_fast32_t numFeatures) override;

    std::vector<value_t> getData() { return this->data; }
};