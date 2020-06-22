#pragma once

#include <kmeans/kmeans/coreset_kmeans.hpp>
#include <kmeans/kmeans/kmeans_impl.hpp>

namespace hpkmeans
{
template <typename T, Parallelism Level = Parallelism::Serial, class DistanceFunc = L2Norm<T>>
class KMeans
{
public:
    KMeans(const std::string& initializer, const std::string& maximizer) :
        p_impl(std::make_unique<KMeansImpl<T, Level, DistanceFunc>>(initializer, maximizer))
    {
    }

    KMeans(const std::string& initializer, const std::string& maximizer, const int coresetRepeats,
           const int32_t sampleSize) :
        p_impl(
          std::make_unique<CoresetKmeans<T, Level, DistanceFunc>>(initializer, maximizer, coresetRepeats, sampleSize))
    {
    }

    const Clusters<T, Level>* const fit(const Matrix<T>* const data, const int32_t& numClusters, const int& numRepeats,
                                        const std::vector<T>* const weights = nullptr)
    {
        return p_impl->fit(data, numClusters, numRepeats, weights);
    }

    const Clusters<T, Level>* const getResults() const { return p_impl->getResults(); }

private:
    std::unique_ptr<KMeansImpl<T, Level, DistanceFunc>> p_impl;
};
}  // namespace hpkmeans