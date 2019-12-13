#include "Writer.hpp"

#include <fstream>

DataSetWriter::DataSetWriter(dataset_t clusters, clustering_t clustering) : clusters(clusters), clustering(clustering)
{
}

void DataSetWriter::writeClusters(std::string filepath)
{
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open specified file");
    }

    int numFeatures = clusters.at(0).size();
    for (int i = 0; i < clusters.size(); i++)
    {
        file.write(reinterpret_cast<char *>(&clusters.at(i)[0]), sizeof(value_t) * numFeatures);
    }
}

void DataSetWriter::writeClustering(std::string filepath)
{
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open specified file");
    }

    file.write(reinterpret_cast<char *>(&clustering.at(0)), sizeof(int) * clustering.size());
}

ClusterWriter::ClusterWriter(clusters_t clusters, clustering_t clustering) : clusters(clusters), clustering(clustering)
{
}

void ClusterWriter::writeClusters(std::string filepath)
{
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open specified file");
    }

    int numFeatures = clusters.at(0).coords.size();
    for (int i = 0; i < clusters.size(); i++)
    {
        file.write(reinterpret_cast<char *>(&clusters.at(i).coords[0]), sizeof(value_t) * numFeatures);
    }
}

void ClusterWriter::writeClustering(std::string filepath)
{
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open specified file");
    }

    file.write(reinterpret_cast<char *>(&clustering.at(0)), sizeof(int) * clustering.size());
}

void ClusterWriter::writeTimes(std::vector<double> times, std::string filepath)
{

    std::ofstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open specified file");
    }

    for (auto &val : times)
    {

        file << val;
        file << std::endl;
    }
}