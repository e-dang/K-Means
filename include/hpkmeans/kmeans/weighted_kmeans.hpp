#pragma once

#include <hpkmeans/kmeans/kmeans_wrapper.hpp>

namespace HPKmeans
{
template <typename precision, typename int_size>
class WeightedKmeansWrapper : public AbstractKmeansWrapper<precision, int_size>
{
public:
    WeightedKmeansWrapper(IKmeansInitializer<precision, int_size>* initializer,
                          IKmeansMaximizer<precision, int_size>* maximizer,
                          IKmeansStateInitializer<precision, int_size>* stateInitializer,
                          std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        AbstractKmeansWrapper<precision, int_size>(initializer, maximizer, stateInitializer, distanceFunc)
    {
    }

    ~WeightedKmeansWrapper() = default;

    std::shared_ptr<ClusterResults<precision, int_size>> fit(const Matrix<precision, int_size>* const data,
                                                             const int_size& numClusters,
                                                             const int& numRestarts) override;

    std::shared_ptr<ClusterResults<precision, int_size>> fit(const Matrix<precision, int_size>* const data,
                                                             const int_size& numClusters, const int& numRestarts,
                                                             const std::vector<precision>* const weights) override;
};

template <typename precision, typename int_size>
std::shared_ptr<ClusterResults<precision, int_size>> WeightedKmeansWrapper<precision, int_size>::fit(
  const Matrix<precision, int_size>* const data, const int_size& numClusters, const int& numRestarts)
{
    std::vector<precision> weights(data->rows(), 1);
    return fit(data, numClusters, numRestarts, &weights);
}

template <typename precision, typename int_size>
std::shared_ptr<ClusterResults<precision, int_size>> WeightedKmeansWrapper<precision, int_size>::fit(
  const Matrix<precision, int_size>* const data, const int_size& numClusters, const int& numRestarts,
  const std::vector<precision>* const weights)
{
    auto kmeansState = this->p_stateInitializer->initializeState(data, weights, this->p_DistanceFunc);
    return this->run(data, numClusters, numRestarts, &kmeansState);
}
}  // namespace HPKmeans