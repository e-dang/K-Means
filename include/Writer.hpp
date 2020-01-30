#pragma once

#include <string>
#include "Definitions.hpp"
#include "DataClasses.hpp"

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
    void writeTimes(std::vector<double> time, std::string);
};

class ClusterDataWriter : public IWriter
{
private:
    ClusterData clusterData;
    int numData;
    int numFeatures;

public:
    ClusterDataWriter(ClusterData clusterData, int numData, int numFeatures) : clusterData(clusterData),
                                                                               numData(numData),
                                                                               numFeatures(numFeatures){};
    ~ClusterDataWriter(){};

    void writeClusters(std::string filepath) override;
    void writeClustering(std::string filepath) override;
};