#include "Writer.hpp"

#include <fstream>

void Writer::writeClusters(clusters_t clusters, std::string filepath)
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

void Writer::writeClustering(clustering_t clustering, std::string filepath)
{
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open specified file");
    }

    file.write(reinterpret_cast<char *>(&clustering.at(0)), sizeof(int) * clustering.size());
}