#include "Utils/Writer.hpp"

#include <fstream>

AbstractWriter::AbstractWriter(Initializer initializer, Maximizer maximizer, CoresetCreator coresetCreator,
                               Parallelism parallelism)
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

void AbstractWriter::writeClusterResults(std::shared_ptr<ClusterResults> clusterResults, const int_fast64_t& time,
                                         std::string& filepath)
{
    writeClusters(&clusterResults->mClusterData.mClusters, filepath);
    writeClustering(&clusterResults->mClusterData.mClustering, filepath);
    writeClusterWeights(&clusterResults->mClusterData.mClusterWeights, filepath);
    writeSqDistances(&clusterResults->mSqDistances, filepath);
    writeRunStats(clusterResults->mError, time, filepath);
}

void AbstractWriter::writeRunStats(const value_t& error, const int_fast64_t& time, std::string& filepath)
{
    auto file = openFile(fileRotator.getUniqueFileName(filepath, "stats"), std::ios::out);

    writeError(error, file);
    writeTime(time, file);
    writeRunParams(file);

    file.close();
}

void AbstractWriter::writeError(const value_t& error, std::ofstream& file) { file << "Error: " << error << std::endl; }

void AbstractWriter::writeTime(const value_t& time, std::ofstream& file) { file << "Time: " << time << std::endl; }

void AbstractWriter::writeRunParams(std::ofstream& file)
{
    for (auto& val : runParams)
    {
        file << val << std::endl;
    }
}

std::ofstream AbstractWriter::openFile(const std::string& filepath, const std::ios::openmode mode)
{
    std::ofstream file(filepath, mode);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file: " << filepath << std::endl;
        exit(1);
    }

    return file;
}

void ClusterResultWriter::writeClusters(Matrix* clusters, std::string& filepath)
{
    auto file = openFile(fileRotator.getUniqueFileName(filepath, "clusters"), std::ios::binary);
    file.write(reinterpret_cast<char*>(clusters->data()),
               sizeof(value_t) * clusters->getNumData() * clusters->getNumFeatures());
    file.close();
}

void ClusterResultWriter::writeClustering(std::vector<int>* clustering, std::string& filepath)
{
    auto file = openFile(fileRotator.getUniqueFileName(filepath, "clustering"), std::ios::binary);
    file.write(reinterpret_cast<char*>(clustering->data()), sizeof(int_fast32_t) * clustering->size());
}

void ClusterResultWriter::writeClusterWeights(std::vector<value_t>* clusterWeights, std::string& filepath)
{
    auto file = openFile(fileRotator.getUniqueFileName(filepath, "clusterWeights"), std::ios::binary);
    file.write(reinterpret_cast<char*>(clusterWeights->data()), sizeof(value_t) * clusterWeights->size());
    file.close();
}

void ClusterResultWriter::writeSqDistances(std::vector<value_t>* sqDistances, std::string& filepath)
{
    auto file = openFile(fileRotator.getUniqueFileName(filepath, "sqDistances"), std::ios::binary);
    file.write(reinterpret_cast<char*>(sqDistances->data()), sizeof(value_t) * sqDistances->size());
    file.close();
}