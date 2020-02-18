#pragma once

#include <fstream>
#include <iomanip>
#include <string>
#include <unordered_map>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
#include "Utils/FileRotator.hpp"
namespace HPKmeans
{
template <typename precision = double, typename int_size = int32_t>
class AbstractWriter
{
protected:
    int m_Digits;
    FileRotator fileRotator;
    std::unordered_map<int, std::string> typeMap;
    std::vector<std::string> runParams;

public:
    AbstractWriter(Initializer initializer, Maximizer maximizer, CoresetCreator coresetCreator, Parallelism parallelism,
                   const int digits = 8) :
        m_Digits(digits)
    {
        typeMap[KPP]       = "K++";
        typeMap[OptKPP]    = "OptK++";
        typeMap[Lloyd]     = "Lloyd";
        typeMap[OptLloyd]  = "OptLloyd";
        typeMap[LWCoreset] = "Coreset";
        typeMap[None]      = "No Coreset";
        typeMap[Serial]    = "Serial";
        typeMap[OMP]       = "OMP";
        typeMap[MPI]       = "MPI";
        typeMap[Hybrid]    = "Hybrid";

        runParams.push_back(typeMap[initializer]);
        runParams.push_back(typeMap[maximizer]);
        runParams.push_back(typeMap[coresetCreator]);
        runParams.push_back(typeMap[parallelism]);
    }

    virtual ~AbstractWriter() = default;

    void writeClusterResults(std::shared_ptr<ClusterResults<precision, int_size>> clusterResults,
                             const int_fast64_t& time, std::string& filepath)
    {
        writeClusters(&clusterResults->clusterData.clusters, filepath);
        writeClustering(&clusterResults->clusterData.clustering, filepath);
        writeClusterWeights(&clusterResults->clusterData.clusterWeights, filepath);
        writeSqDistances(&clusterResults->sqDistances, filepath);
        writeRunStats(clusterResults->error, time, filepath);
    }

    void writeRunStats(const precision& error, const int_fast64_t& time, std::string& filepath)
    {
        auto file = this->openFile(this->fileRotator.getUniqueFileName(filepath, "stats"), std::ios::out);

        writeError(error, file);
        writeTime(time, file);
        writeRunParams(file);

        file.close();
    }

    void writeError(const precision& error, std::ofstream& file)
    {
        file << std::fixed << std::setprecision(m_Digits) << "Error: " << error << std::endl;
    }

    void writeTime(const precision& time, std::ofstream& file)
    {
        file << std::fixed << std::setprecision(m_Digits) << "Time: " << time << std::endl;
    }

    void writeRunParams(std::ofstream& file)
    {
        for (auto& val : runParams)
        {
            file << val << std::endl;
        }
    }

    virtual void writeClusterWeights(std::vector<precision>* clusterWeights, std::string& filepath) = 0;

    virtual void writeClusters(Matrix<precision, int_size>* clusters, std::string& filepath) = 0;

    virtual void writeClustering(std::vector<int_size>* clustering, std::string& filepath) = 0;

    virtual void writeSqDistances(std::vector<precision>* sqDistances, std::string& filepath) = 0;

protected:
    std::ofstream openFile(const std::string& filepath, const std::ios::openmode mode)
    {
        std::ofstream file(filepath, mode);
        if (!file.is_open())
        {
            std::cerr << "Unable to open file: " << filepath << std::endl;
            exit(1);
        }

        return file;
    }
};

template <typename precision = double, typename int_size = int32_t>
class ClusterResultWriter : public AbstractWriter<precision, int_size>
{
public:
    ClusterResultWriter(Initializer initializer, Maximizer maximizer, CoresetCreator coresetCreator,
                        Parallelism parallelism) :
        AbstractWriter<precision, int_size>(initializer, maximizer, coresetCreator, parallelism)
    {
    }

    ~ClusterResultWriter() = default;

    void writeClusters(Matrix<precision, int_size>* clusters, std::string& filepath) override
    {
        auto file = this->openFile(this->fileRotator.getUniqueFileName(filepath, "clusters"), std::ios::binary);
        file.write(reinterpret_cast<char*>(clusters->data()), sizeof(precision) * clusters->size() * clusters->cols());
        file.close();
    }

    void writeClustering(std::vector<int_size>* clustering, std::string& filepath) override
    {
        auto file = this->openFile(this->fileRotator.getUniqueFileName(filepath, "clustering"), std::ios::binary);
        file.write(reinterpret_cast<char*>(clustering->data()), sizeof(int_size) * clustering->size());
        file.close();
    }

    void writeClusterWeights(std::vector<precision>* clusterWeights, std::string& filepath) override
    {
        auto file = this->openFile(this->fileRotator.getUniqueFileName(filepath, "clusterWeights"), std::ios::binary);
        file.write(reinterpret_cast<char*>(clusterWeights->data()), sizeof(precision) * clusterWeights->size());
        file.close();
    }

    void writeSqDistances(std::vector<precision>* sqDistances, std::string& filepath) override
    {
        auto file = this->openFile(this->fileRotator.getUniqueFileName(filepath, "sqDistances"), std::ios::binary);
        file.write(reinterpret_cast<char*>(sqDistances->data()), sizeof(precision) * sqDistances->size());
        file.close();
    }
};
}  // namespace HPKmeans