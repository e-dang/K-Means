#pragma once

#include <string>
#include "Definitions.hpp"

class IWriter
{
public:
    virtual void writeClusters(std::string filepath) = 0;
    virtual void writeClustering(std::string filepath) = 0;
};

class DataSetWriter : public IWriter
{
private:
    dataset_t clusters;
    clustering_t clustering;

public:
    DataSetWriter(dataset_t clusters, clustering_t clustering);
    ~DataSetWriter(){};

    void writeClusters(std::string filepath) override;
    void writeClustering(std::string filepath) override;
};

class ClusterWriter : public IWriter
{
private:
    clusters_t clusters;
    clustering_t clustering;

public:
    ClusterWriter(clusters_t clusters, clustering_t clustering);
    ~ClusterWriter(){};

    void writeClusters(std::string filepath) override;
    void writeClustering(std::string filepath) override;
};