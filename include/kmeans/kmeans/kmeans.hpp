#pragma once

#include <kmeans/distances.hpp>
#include <kmeans/initializers/initializers.hpp>
#include <kmeans/maximizers/maximizers.hpp>
#include <memory>
#include <string>

namespace hpkmeans
{
template <typename T, Parallelism Level = Parallelism::Serial, class DistanceFunc = L2Norm<T>>
class KMeans
{
public:
    KMeans(const std::string& initializer, const std::string& maximizer) :
        p_initializer(createInitializer<T, Level, DistanceFunc>(initializer)),
        p_maximizer(createMaximizer<T, Level, DistanceFunc>(maximizer))
    {
    }

    virtual ~KMeans() = default;

    const Clusters<T, Level>* const fit(const Matrix<T>* const data, const int32_t& numClusters, const int& numRepeats)
    {
        Clusters<T, Level> clusters(numClusters, data);

        for (int i = 0; i < numRepeats; ++i)
        {
            p_initializer->initialize(data, &clusters);
            p_maximizer->maximize(data, &clusters);
            compareResults(clusters, m_bestClusters);
            clusters.clear();
        }

        return getResults();
    }

    const Clusters<T, Level>* const getResults() const { return &m_bestClusters; }

protected:
    void compareResults(const Clusters<T, Level>& candidateClusters, Clusters<T, Level>& bestClusters)
    {
        if (candidateClusters < bestClusters)
            bestClusters = candidateClusters;
    }

protected:
    Clusters<T, Level> m_bestClusters;

private:
    std::unique_ptr<IInitializer<T, Level>> p_initializer;
    std::unique_ptr<IMaximizer<T, Level>> p_maximizer;
};
}  // namespace hpkmeans