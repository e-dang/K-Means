#include "Reader.hpp"
#include <iostream>
#include <fstream>

void CReader::read(std::string filepath, int numData, int numFeatures)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open specified file");
    }

    this->data = new value_t[numData * numFeatures];
    for (int i = 0; i < numData * numFeatures; i++)
    {
        file.read(reinterpret_cast<char *>(&data[i]), sizeof(value_t));
    }
}

void DataSetReader::read(std::string filepath, int numData, int numFeatures)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open specified file");
    }

    data = dataset_t(numData);
    for (int i = 0; i < numData; i++)
    {
        data.at(i).resize(numFeatures);
        file.read(reinterpret_cast<char *>(&data.at(i)[0]), sizeof(value_t) * numFeatures);
    }
}