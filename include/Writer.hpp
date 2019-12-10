#pragma once

#include <string>
#include "Definitions.hpp"

class Writer
{
private:
public:
    Writer(){};
    ~Writer(){};

    void writeClusters(clusters_t clusters, std::string filepath);
    void writeClustering(clustering_t clustering, std::string filepath);
};