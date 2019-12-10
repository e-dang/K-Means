#pragma once

#include <string>
#include "Definitions.hpp"

class IReader
{
public:
    virtual void read(std::string filepath, int numData, int numFeatures) = 0;
};

class CReader : public IReader
{
private:
    value_t *data;

public:
    CReader(){};
    ~CReader(){};

    void read(std::string filepath, int numData, int numFeatures) override;

    value_t *getData() { return this->data; }
};

class DataSetReader : public IReader
{
private:
    dataset_t data;

public:
    DataSetReader(){};
    ~DataSetReader(){};

    void read(std::string filepath, int numData, int numFeatures) override;

    dataset_t getData() { return this->data; }
};