#pragma once

#include <string>
#include <unordered_map>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
#include "Utils/FileRotator.hpp"

class AbstractWriter
{
protected:
    FileRotator fileRotator;
    std::unordered_map<int, std::string> typeMap;
    std::vector<std::string> runParams;

public:
    AbstractWriter(Initializer initializer, Maximizer maximizer, CoresetCreator coresetCreator,
                   Parallelism parallelism);

    virtual ~AbstractWriter() {}

    void writeClusterResults(std::shared_ptr<ClusterResults> clusterResults, const int_fast64_t& time,
                             std::string& filepath);

    void writeRunStats(const value_t& error, const int_fast64_t& time, std::string& filepath);

    void writeError(const value_t& error, std::ofstream& file);

    void writeTime(const value_t& time, std::ofstream& file);

    void writeRunParams(std::ofstream& file);

    virtual void writeClusterWeights(std::vector<value_t>* clusterWeights, std::string& filepath) = 0;

    virtual void writeClusters(Matrix* clusters, std::string& filepath) = 0;

    virtual void writeClustering(std::vector<int>* clustering, std::string& filepath) = 0;

    virtual void writeSqDistances(std::vector<value_t>* sqDistances, std::string& filepath) = 0;

protected:
    std::ofstream openFile(const std::string& filepath, const std::ios::openmode mode);
};

class ClusterResultWriter : public AbstractWriter
{
public:
    ClusterResultWriter(Initializer initializer, Maximizer maximizer, CoresetCreator coresetCreator,
                        Parallelism parallelism) :
        AbstractWriter(initializer, maximizer, coresetCreator, parallelism)
    {
    }

    ~ClusterResultWriter() {}

    void writeClusters(Matrix* clusters, std::string& filepath) override;

    void writeClustering(std::vector<int>* clustering, std::string& filepath) override;

    void writeClusterWeights(std::vector<value_t>* clusterWeights, std::string& filepath) override;

    void writeSqDistances(std::vector<value_t>* sqDistances, std::string& filepath) override;
};