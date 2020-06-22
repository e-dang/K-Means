#pragma once

#include <hpkmeans/distances.hpp>
#include <hpkmeans/initializers/initializers.hpp>
#include <hpkmeans/maximizers/maximizers.hpp>
#include <memory>
#include <string>

namespace hpkmeans
{
template <typename T, Parallelism Level, class DistanceFunc>
class KMeansImpl
{
public:
    KMeansImpl(const std::string& initializer, const std::string& maximizer) :
        p_initializer(createInitializer<T, Level, DistanceFunc>(initializer)),
        p_maximizer(createMaximizer<T, Level, DistanceFunc>(maximizer))
    {
    }

    virtual ~KMeansImpl() = default;

    virtual const Clusters<T, Level>* const fit(const Matrix<T>* const data, const int32_t& numClusters,
                                                const int& numRepeats, const std::vector<T>* const weights = nullptr)
    {
        Clusters<T, Level> clusters(numClusters, data, weights);

        for (int i = 0; i < numRepeats; ++i)
        {
            p_initializer->initialize(data, &clusters);
            p_maximizer->maximize(data, &clusters);
            compareResults(clusters, m_bestClusters);
            clusters.clear();
        }

        return KMeansImpl::getResults();
    }

    virtual const Clusters<T, Level>* const getResults() const { return &m_bestClusters; }

protected:
    void compareResults(const Clusters<T, Level>& candidateClusters, Clusters<T, Level>& bestClusters)
    {
        if (candidateClusters < bestClusters)
            bestClusters = candidateClusters;
    }

private:
    Clusters<T, Level> m_bestClusters;
    std::unique_ptr<IInitializer<T, Level>> p_initializer;
    std::unique_ptr<IMaximizer<T, Level>> p_maximizer;
};
}  // namespace hpkmeans
