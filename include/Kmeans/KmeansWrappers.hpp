#pragma once

#include <memory>
#include <numeric>

#include "Containers/Definitions.hpp"
#include "Containers/KmeansState.hpp"
#include "KmeansAlgorithms/CoresetCreator.hpp"
#include "KmeansAlgorithms/KmeansAlgorithms.hpp"
#include "Strategies/CoresetClusteringFinisher.hpp"
#include "Strategies/KmeansStateInitializer.hpp"
#include "Utils/DistanceFunctors.hpp"
#include "mpi.h"
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
    std::unique_ptr<AbstractKmeansInitializer<precision, int_size>> p_Initializer;
    std::unique_ptr<AbstractKmeansMaximizer<precision, int_size>> p_Maximizer;
    std::unique_ptr<IKmeansStateInitializer<precision, int_size>> p_stateInitializer;
    std::shared_ptr<IDistanceFunctor<precision>> p_DistanceFunc;

public:
    AbstractKmeansWrapper(IKmeansStateInitializer<precision, int_size>* stateInitializer,
                          std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        p_stateInitializer(stateInitializer), p_DistanceFunc(distanceFunc)
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
    AbstractKmeansWrapper(AbstractKmeansInitializer<precision, int_size>* initializer,
                          AbstractKmeansMaximizer<precision, int_size>* maximizer,
                          IKmeansStateInitializer<precision, int_size>* stateInitializer,
                          std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        p_Initializer(initializer),
        p_Maximizer(maximizer),
        p_stateInitializer(stateInitializer),
        p_DistanceFunc(distanceFunc)
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
class WeightedKmeansWrapper : public AbstractKmeansWrapper<precision, int_size>
{
public:
    WeightedKmeansWrapper(AbstractKmeansInitializer<precision, int_size>* initializer,
                          AbstractKmeansMaximizer<precision, int_size>* maximizer,
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