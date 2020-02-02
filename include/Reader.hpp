#pragma once

#include <string>

#include "DataClasses.hpp"
#include "Definitions.hpp"

class IReader
{
public:
    virtual void read(std::string filepath, int numData, int numFeatures) = 0;
};

class VectorReader : public IReader
{
private:
    std::vector<value_t> data;

public:
    VectorReader(){};
    ~VectorReader(){};

    void read(std::string filepath, int numData, int numFeatures) override;

    std::vector<value_t> getData() { return this->data; }
};

class MPIReader : public IReader
{
private:
    std::vector<value_t> data;

public:
    MPIReader(){};
    ~MPIReader(){};

    void read(std::string filepath, int numData, int numFeatures) override;

    std::vector<value_t> getData() { return this->data; }
};