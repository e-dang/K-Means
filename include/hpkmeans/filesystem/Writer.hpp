#pragma once

#include <fstream>
#include <hpkmeans/data_types/cluster_results.hpp>
#include <hpkmeans/data_types/enums.hpp>
#include <hpkmeans/filesystem/FileRotator.hpp>
#include <iomanip>
#include <string>
#include <unordered_map>

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
                   const int digits = 8);

    virtual ~AbstractWriter() = default;

    void writeClusterResults(std::shared_ptr<ClusterResults<precision, int_size>> clusterResults,
                             const int_fast64_t& time, std::string& filepath);

    void writeRunStats(const precision& error, const int_fast64_t& time, std::string& filepath);

    void writeError(const precision& error, std::ofstream& file);

    void writeTime(const precision& time, std::ofstream& file);

    void writeRunParams(std::ofstream& file);

    virtual void writeClusterWeights(std::vector<precision>* clusterWeights, std::string& filepath) = 0;

    virtual void writeClusters(Matrix<precision, int_size>* clusters, std::string& filepath) = 0;

    virtual void writeClustering(std::vector<int_size>* clustering, std::string& filepath) = 0;

    virtual void writeSqDistances(std::vector<precision>* sqDistances, std::string& filepath) = 0;

protected:
    std::ofstream openFile(const std::string& filepath, const std::ios::openmode mode);
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

    void writeClusters(Matrix<precision, int_size>* clusters, std::string& filepath) override;

    void writeClustering(std::vector<int_size>* clustering, std::string& filepath) override;

    void writeClusterWeights(std::vector<precision>* clusterWeights, std::string& filepath) override;

    void writeSqDistances(std::vector<precision>* sqDistances, std::string& filepath) override;
};

template <typename precision, typename int_size>
AbstractWriter<precision, int_size>::AbstractWriter(Initializer initializer, Maximizer maximizer,
                                                    CoresetCreator coresetCreator, Parallelism parallelism,
                                                    const int digits) :
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

template <typename precision, typename int_size>
void AbstractWriter<precision, int_size>::writeClusterResults(
  std::shared_ptr<ClusterResults<precision, int_size>> clusterResults, const int_fast64_t& time, std::string& filepath)
{
    writeClusters(clusterResults->clusters.get(), filepath);
    writeClustering(clusterResults->clustering.get(), filepath);
    writeClusterWeights(clusterResults->clusterWeights.get(), filepath);
    writeSqDistances(clusterResults->sqDistances.get(), filepath);
    writeRunStats(clusterResults->error, time, filepath);
}

template <typename precision, typename int_size>
void AbstractWriter<precision, int_size>::writeRunStats(const precision& error, const int_fast64_t& time,
                                                        std::string& filepath)
{
    auto file = this->openFile(this->fileRotator.getUniqueFileName(filepath, "stats"), std::ios::out);

    writeError(error, file);
    writeTime(time, file);
    writeRunParams(file);

    file.close();
}

template <typename precision, typename int_size>
void AbstractWriter<precision, int_size>::writeError(const precision& error, std::ofstream& file)
{
    auto ss = file.precision();
    file << std::fixed << std::setprecision(m_Digits) << "Error: " << error << std::endl;
    file << std::defaultfloat << std::setprecision(ss);
}

template <typename precision, typename int_size>
void AbstractWriter<precision, int_size>::writeTime(const precision& time, std::ofstream& file)
{
    file << "Time: " << time << std::endl;
}

template <typename precision, typename int_size>
void AbstractWriter<precision, int_size>::writeRunParams(std::ofstream& file)
{
    for (auto& val : runParams)
    {
        file << val << std::endl;
    }
}

template <typename precision, typename int_size>
std::ofstream AbstractWriter<precision, int_size>::openFile(const std::string& filepath, const std::ios::openmode mode)
{
    std::ofstream file(filepath, mode);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file: " << filepath << std::endl;
        exit(1);
    }

    return file;
}

template <typename precision, typename int_size>
void ClusterResultWriter<precision, int_size>::writeClusters(Matrix<precision, int_size>* clusters,
                                                             std::string& filepath)
{
    auto file = this->openFile(this->fileRotator.getUniqueFileName(filepath, "clusters"), std::ios::binary);
    file.write(reinterpret_cast<char*>(clusters->data()), sizeof(precision) * clusters->size() * clusters->cols());
    file.close();
}

template <typename precision, typename int_size>
void ClusterResultWriter<precision, int_size>::writeClustering(std::vector<int_size>* clustering, std::string& filepath)
{
    auto file = this->openFile(this->fileRotator.getUniqueFileName(filepath, "clustering"), std::ios::binary);
    file.write(reinterpret_cast<char*>(clustering->data()), sizeof(int_size) * clustering->size());
    file.close();
}

template <typename precision, typename int_size>
void ClusterResultWriter<precision, int_size>::writeClusterWeights(std::vector<precision>* clusterWeights,
                                                                   std::string& filepath)
{
    auto file = this->openFile(this->fileRotator.getUniqueFileName(filepath, "clusterWeights"), std::ios::binary);
    file.write(reinterpret_cast<char*>(clusterWeights->data()), sizeof(precision) * clusterWeights->size());
    file.close();
}

template <typename precision, typename int_size>
void ClusterResultWriter<precision, int_size>::writeSqDistances(std::vector<precision>* sqDistances,
                                                                std::string& filepath)
{
    auto file = this->openFile(this->fileRotator.getUniqueFileName(filepath, "sqDistances"), std::ios::binary);
    file.write(reinterpret_cast<char*>(sqDistances->data()), sizeof(precision) * sqDistances->size());
    file.close();
}
}  // namespace HPKmeans