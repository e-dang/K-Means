#include "Utils/Writer.hpp"

#include <fstream>

void ClusterDataWriter::writeClusters(std::string filepath)
{
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open specified file");
    }

    file.write(reinterpret_cast<char*>(clusterData.mClusters.data()),
               sizeof(value_t) * clusterData.mClusters.getNumData() * numFeatures);
}

void ClusterDataWriter::writeClustering(std::string filepath)
{
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open specified file");
    }

    file.write(reinterpret_cast<char*>(clusterData.mClustering.data()), sizeof(int_fast32_t) * numData);
}

void ClusterDataWriter::writeTimes(std::vector<double> times, std::string filepath)
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open specified file");
    }

    for (auto& val : times)
    {
        file << val;
        file << std::endl;
    }
}