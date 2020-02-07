#pragma once

#include <string>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"

class IWriter
{
public:
    virtual void writeClusters(std::string filepath)   = 0;
    virtual void writeClustering(std::string filepath) = 0;
};

class ClusterDataWriter : public IWriter
{
private:
    ClusterData clusterData;
    int_fast32_t numData;
    int_fast32_t numFeatures;

public:
    ClusterDataWriter(ClusterData clusterData, int_fast32_t numData, int_fast32_t numFeatures) :
        clusterData(clusterData), numData(numData), numFeatures(numFeatures){};
    ~ClusterDataWriter(){};

    void writeClusters(std::string filepath) override;
    void writeClustering(std::string filepath) override;
    void writeTimes(std::vector<double> times, std::string filepath);
};