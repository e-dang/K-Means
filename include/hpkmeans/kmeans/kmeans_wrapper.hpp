#pragma once

#include <mpi.h>

#include <hpkmeans/algorithms/kmeans_algorithm.hpp>
#include <hpkmeans/data_types/cluster_results.hpp>
#include <hpkmeans/factories/state_factories.hpp>
#include <memory>
#include <numeric>

namespace HPKmeans
{
/**
 * @brief Abstract class that defines the interface for using a Kmeans class, which wraps an initialization and
 *        maximization algorithm together, along with a distance metric functor in order to cluster data. In addition
 *        this class also defines the member variables, setters, getters, and helper functions that each Kmeans
 *        concretion will need to function.
 */
template <typename precision, typename int_size>
class AbstractKmeansWrapper
{
protected:
    std::unique_ptr<IKmeansInitializer<precision, int_size>> p_Initializer;
    std::unique_ptr<IKmeansMaximizer<precision, int_size>> p_Maximizer;
    std::unique_ptr<IKmeansStateFactory<precision, int_size>> p_stateFactory;
    std::shared_ptr<IDistanceFunctor<precision>> p_DistanceFunc;

public:
    AbstractKmeansWrapper(IKmeansStateFactory<precision, int_size>* stateFactory,
                          std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        p_stateFactory(stateFactory), p_DistanceFunc(distanceFunc)
    {
    }

    /**
     * @brief Constructor for AbstractKmeans.
     *
     * @param initializer - A pointer to a class implementing a Kmeans initialization algorithm such as K++.
     * @param maximizer - A pointer to a class implementing a Kmeans maximization algorithm such as lloyd's algorithm.
     * @param distanceFunc - A pointer to a functor class used to calculate the distance between points, such as the
     *                       euclidean distance.
     */
    AbstractKmeansWrapper(IKmeansInitializer<precision, int_size>* initializer,
                          IKmeansMaximizer<precision, int_size>* maximizer,
                          IKmeansStateFactory<precision, int_size>* stateFactory,
                          std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        p_Initializer(initializer), p_Maximizer(maximizer), p_stateFactory(stateFactory), p_DistanceFunc(distanceFunc)
    {
    }

    virtual ~AbstractKmeansWrapper() = default;

    /**
     * @brief Overloaded interface for the top level function that initiates the clustering process, where the weights
     *        of each datapoint is unspecified. This function should form a vector of uniform weights and pass it to
     *        the fit() function that takes weights as parameter, but the specific implementation is left up to the
     *        concretion of this class.
     *
     * @param data - The data to be clustered.
     * @param numClusters - The number of clusters to cluster the data into.
     * @param numRestarts - The number of times to repeat the clustering process.
     */
    virtual std::shared_ptr<ClusterResults<precision, int_size>> fit(const Matrix<precision, int_size>* const data,
                                                                     const int_size& numClusters,
                                                                     const int& numRestarts) = 0;

    /**
     * @brief Interface for the top level function that initiates the clustering process.
     *
     * @param data - The data to be clustered.
     * @param numClusters - The number of clusters to cluster the data into.
     * @param numRestarts - The number of times to repeat the clustering process.
     * @param weights - The weights for each datapoint in the matrix.
     */
    virtual std::shared_ptr<ClusterResults<precision, int_size>> fit(const Matrix<precision, int_size>* const data,
                                                                     const int_size& numClusters,
                                                                     const int& numRestarts,
                                                                     const std::vector<precision>* const weights) = 0;

    /**
     * @brief Set the distanceFunc member variable.
     *
     * @param distanceFunc - A pointer to an implementation of the IDistanceFunctor class.
     */
    void setDistanceFunc(IDistanceFunctor<precision>* distanceFunc) { p_DistanceFunc.reset(distanceFunc); }

protected:
    std::shared_ptr<ClusterResults<precision, int_size>> run(const Matrix<precision, int_size>* const data,
                                                             const int_size& numClusters, const int& numRestarts,
                                                             KmeansState<precision, int_size>* const kmeansState);
};

template <typename precision, typename int_size>
std::shared_ptr<ClusterResults<precision, int_size>> AbstractKmeansWrapper<precision, int_size>::run(
  const Matrix<precision, int_size>* const data, const int_size& numClusters, const int& numRestarts,
  KmeansState<precision, int_size>* const kmeansState)
{
    std::shared_ptr<ClusterResults<precision, int_size>> clusterResults =
      std::make_shared<ClusterResults<precision, int_size>>();

    p_Initializer->setState(kmeansState);
    p_Maximizer->setState(kmeansState);

    for (int i = 0; i < numRestarts; ++i)
    {
        kmeansState->resetClusterData(numClusters, 1.0);

        p_Initializer->initialize();
        p_Maximizer->maximize();

        kmeansState->compareResults(clusterResults);
    }

    return clusterResults;
}
}  // namespace HPKmeans