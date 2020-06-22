#pragma once

#include <fstream>
#include <iomanip>
#include <hpkmeans/filesystem/file_rotator.hpp>
#include <hpkmeans/types/clusters.hpp>
#include <hpkmeans/types/parallelism.hpp>
#include <matrix/matrix.hpp>
#include <string>

namespace hpkmeans
{
template <typename T, Parallelism Level>
class AbstractWriter
{
protected:
    int m_digits;
    FileRotator m_fileRotator;
    std::vector<std::string> m_runParams;

public:
    AbstractWriter(Parallelism parallelism, const int digits = 8);

    virtual ~AbstractWriter() = default;

    void writeClusterResults(const Clusters<T, Level>* clusterResults, const int_fast64_t& time, std::string& filepath);

    void writeRunStats(const T& error, const int_fast64_t& time, std::string& filepath);

    void writeError(const T& error, std::ofstream& file);

    void writeTime(const T& time, std::ofstream& file);

    void writeRunParams(std::ofstream& file);

    virtual void writeClusters(const Matrix<T>* clusters, std::string& filepath) = 0;

    virtual void writeClustering(const VectorView<int32_t>* clustering, std::string& filepath) = 0;

protected:
    std::ofstream openFile(const std::string& filepath, const std::ios::openmode mode);
};

template <typename T, Parallelism Level>
class ClusterResultWriter : public AbstractWriter<T, Level>
{
public:
    ClusterResultWriter(Parallelism parallelism) : AbstractWriter<T, Level>(parallelism) {}

    ~ClusterResultWriter() = default;

    void writeClusters(const Matrix<T>* clusters, std::string& filepath) override;

    void writeClustering(const VectorView<int32_t>* clustering, std::string& filepath) override;
};

template <typename T, Parallelism Level>
AbstractWriter<T, Level>::AbstractWriter(Parallelism parallelism, const int digits) : m_digits(digits)
{
    m_runParams.push_back(parallelismToString(parallelism));
}

template <typename T, Parallelism Level>
void AbstractWriter<T, Level>::writeClusterResults(const Clusters<T, Level>* clusterResults, const int_fast64_t& time,
                                                   std::string& filepath)
{
    writeClusters(clusterResults->getCentroids(), filepath);
    writeClustering(clusterResults->assignments(), filepath);
    writeRunStats(clusterResults->getError(), time, filepath);
}

template <typename T, Parallelism Level>
void AbstractWriter<T, Level>::writeRunStats(const T& error, const int_fast64_t& time, std::string& filepath)
{
    auto file = this->openFile(this->m_fileRotator.getUniqueFileName(filepath, "stats"), std::ios::out);

    writeError(error, file);
    writeTime(time, file);
    writeRunParams(file);

    file.close();
}

template <typename T, Parallelism Level>
void AbstractWriter<T, Level>::writeError(const T& error, std::ofstream& file)
{
    auto ss = file.precision();
    file << std::fixed << std::setprecision(m_digits) << "Error: " << error << std::endl;
    file << std::defaultfloat << std::setprecision(ss);
}

template <typename T, Parallelism Level>
void AbstractWriter<T, Level>::writeTime(const T& time, std::ofstream& file)
{
    file << "Time: " << time << std::endl;
}

template <typename T, Parallelism Level>
void AbstractWriter<T, Level>::writeRunParams(std::ofstream& file)
{
    for (auto& val : m_runParams)
    {
        file << val << std::endl;
    }
}

template <typename T, Parallelism Level>
std::ofstream AbstractWriter<T, Level>::openFile(const std::string& filepath, const std::ios::openmode mode)
{
    std::ofstream file(filepath, mode);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file: " << filepath << std::endl;
        exit(1);
    }

    return file;
}

template <typename T, Parallelism Level>
void ClusterResultWriter<T, Level>::writeClusters(const Matrix<T>* clusters, std::string& filepath)
{
    auto file = this->openFile(this->m_fileRotator.getUniqueFileName(filepath, "clusters"), std::ios::binary);
    file.write(clusters->serialize(), clusters->bytes());
    file.close();
}

template <typename T, Parallelism Level>
void ClusterResultWriter<T, Level>::writeClustering(const VectorView<int32_t>* clustering, std::string& filepath)
{
    auto file = this->openFile(this->m_fileRotator.getUniqueFileName(filepath, "clustering"), std::ios::binary);
    file.write(reinterpret_cast<const char*>(clustering->data()), sizeof(int32_t) * clustering->size());
    file.close();
}
}  // namespace hpkmeans