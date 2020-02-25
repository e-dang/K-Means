#pragma once

#include <hpkmeans/kmeans/kmeans_wrapper.hpp>

namespace HPKmeans
{
template <typename precision, typename int_size>
class CoresetKmeansWrapper : public AbstractKmeansWrapper<precision, int_size>
{
private:
    int_size m_SampleSize;
    std::unique_ptr<AbstractKmeansWrapper<precision, int_size>> p_Kmeans;
    std::unique_ptr<AbstractCoresetCreator<precision, int_size>> p_CoresetCreator;
    std::unique_ptr<AbstractCoresetClusteringFinisher<precision, int_size>> p_Finisher;

public:
    CoresetKmeansWrapper(const int_size& sampleSize, AbstractKmeansWrapper<precision, int_size>* kmeans,
                         AbstractCoresetCreator<precision, int_size>* coresetCreator,
                         AbstractCoresetClusteringFinisher<precision, int_size>* finisher,
                         IKmeansStateInitializer<precision, int_size>* stateInitializer,
                         std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        AbstractKmeansWrapper<precision, int_size>(stateInitializer, distanceFunc),
        m_SampleSize(sampleSize),
        p_Kmeans(kmeans),
        p_CoresetCreator(coresetCreator),
        p_Finisher(finisher)
    {
    }

    ~CoresetKmeansWrapper() = default;

    std::shared_ptr<ClusterResults<precision, int_size>> fit(const Matrix<precision, int_size>* const data,
                                                             const int_size& numClusters,
                                                             const int& numRestarts) override;

    std::shared_ptr<ClusterResults<precision, int_size>> fit(const Matrix<precision, int_size>* const data,
                                                             const int_size& numClusters, const int& numRestarts,
                                                             const std::vector<precision>* const weights) override;
};

template <typename precision, typename int_size>
std::shared_ptr<ClusterResults<precision, int_size>> CoresetKmeansWrapper<precision, int_size>::fit(
  const Matrix<precision, int_size>* const data, const int_size& numClusters, const int& numRestarts)
{
    auto kmeansState = this->p_stateInitializer->initializeState(data, nullptr, this->p_DistanceFunc);

    p_CoresetCreator->setState(&kmeansState);
    auto coreset = p_CoresetCreator->createCoreset();

    auto clusterResults = p_Kmeans->fit(&coreset.data, numClusters, numRestarts, &coreset.weights);

    clusterResults->error = -1.0;
    // TODO:has to allocate memory and then move for clustering and clusterWeights and sqDistances...if they sahre
    // pointer could be faster
    kmeansState.setClusters(clusterResults->clusters);
    kmeansState.resetClusterData(numClusters);

    p_Finisher->finishClustering(&kmeansState);
    kmeansState.compareResults(clusterResults);

    return clusterResults;
}

template <typename precision, typename int_size>
std::shared_ptr<ClusterResults<precision, int_size>> CoresetKmeansWrapper<precision, int_size>::fit(
  const Matrix<precision, int_size>* const data, const int_size& numClusters, const int& numRestarts,
  const std::vector<precision>* const weights)
{
    throw std::runtime_error("Should not be calling this func.");
    return nullptr;
}
}  // namespace HPKmeans