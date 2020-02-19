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
class AbstractKmeans
{
protected:
    std::unique_ptr<AbstractKmeansInitializer<precision, int_size>> p_Initializer;
    std::unique_ptr<AbstractKmeansMaximizer<precision, int_size>> p_Maximizer;
    std::unique_ptr<IKmeansStateInitializer<precision, int_size>> p_KmeansStateInitializer;
    std::shared_ptr<IDistanceFunctor<precision>> p_DistanceFunc;

public:
    AbstractKmeans(IKmeansStateInitializer<precision, int_size>* dataCreator,
                   std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        p_KmeansStateInitializer(dataCreator), p_DistanceFunc(distanceFunc)
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
    AbstractKmeans(AbstractKmeansInitializer<precision, int_size>* initializer,
                   AbstractKmeansMaximizer<precision, int_size>* maximizer,
                   IKmeansStateInitializer<precision, int_size>* dataCreator,
                   std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        p_Initializer(initializer),
        p_Maximizer(maximizer),
        p_KmeansStateInitializer(dataCreator),
        p_DistanceFunc(distanceFunc)
    {
    }

    virtual ~AbstractKmeans() = default;

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
class WeightedKmeans : public AbstractKmeans<precision, int_size>
{
public:
    WeightedKmeans(AbstractKmeansInitializer<precision, int_size>* initializer,
                   AbstractKmeansMaximizer<precision, int_size>* maximizer,
                   IKmeansStateInitializer<precision, int_size>* dataCreator,
                   std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        AbstractKmeans<precision, int_size>(initializer, maximizer, dataCreator, distanceFunc)
    {
    }

    ~WeightedKmeans() = default;

    std::shared_ptr<ClusterResults<precision, int_size>> fit(const Matrix<precision, int_size>* const data,
                                                             const int_size& numClusters,
                                                             const int& numRestarts) override;

    std::shared_ptr<ClusterResults<precision, int_size>> fit(const Matrix<precision, int_size>* const data,
                                                             const int_size& numClusters, const int& numRestarts,
                                                             const std::vector<precision>* const weights) override;
};

template <typename precision, typename int_size>
class CoresetKmeans : public AbstractKmeans<precision, int_size>
{
private:
    int_size mSampleSize;
    std::unique_ptr<AbstractKmeans<precision, int_size>> pKmeans;
    std::unique_ptr<AbstractCoresetCreator<precision, int_size>> pCreator;
    std::unique_ptr<AbstractCoresetClusteringFinisher<precision, int_size>> pFinisher;

public:
    CoresetKmeans(const int_size& sampleSize, AbstractKmeans<precision, int_size>* kmeans,
                  AbstractCoresetCreator<precision, int_size>* creator,
                  AbstractCoresetClusteringFinisher<precision, int_size>* finisher,
                  IKmeansStateInitializer<precision, int_size>* dataCreator,
                  std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        AbstractKmeans<precision, int_size>(dataCreator, distanceFunc),
        mSampleSize(sampleSize),
        pKmeans(kmeans),
        pCreator(creator),
        pFinisher(finisher)
    {
    }

    ~CoresetKmeans() = default;

    std::shared_ptr<ClusterResults<precision, int_size>> fit(const Matrix<precision, int_size>* const data,
                                                             const int_size& numClusters,
                                                             const int& numRestarts) override;

    std::shared_ptr<ClusterResults<precision, int_size>> fit(const Matrix<precision, int_size>* const data,
                                                             const int_size& numClusters, const int& numRestarts,
                                                             const std::vector<precision>* const weights) override;
};

template <typename precision, typename int_size>
std::shared_ptr<ClusterResults<precision, int_size>> AbstractKmeans<precision, int_size>::run(
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
std::shared_ptr<ClusterResults<precision, int_size>> WeightedKmeans<precision, int_size>::fit(
  const Matrix<precision, int_size>* const data, const int_size& numClusters, const int& numRestarts)
{
    std::vector<precision> weights(data->rows(), 1);
    return fit(data, numClusters, numRestarts, &weights);
}

template <typename precision, typename int_size>
std::shared_ptr<ClusterResults<precision, int_size>> WeightedKmeans<precision, int_size>::fit(
  const Matrix<precision, int_size>* const data, const int_size& numClusters, const int& numRestarts,
  const std::vector<precision>* const weights)
{
    auto kmeansState = this->p_KmeansStateInitializer->create(data, weights, this->p_DistanceFunc);
    return this->run(data, numClusters, numRestarts, &kmeansState);
}

template <typename precision, typename int_size>
std::shared_ptr<ClusterResults<precision, int_size>> CoresetKmeans<precision, int_size>::fit(
  const Matrix<precision, int_size>* const data, const int_size& numClusters, const int& numRestarts)
{
    auto kmeansState = this->p_KmeansStateInitializer->create(data, nullptr, this->p_DistanceFunc);

    pCreator->setState(&kmeansState);
    auto coreset = pCreator->createCoreset();

    auto clusterResults = pKmeans->fit(&coreset.data, numClusters, numRestarts, &coreset.weights);

    clusterResults->error = -1.0;
    // TODO:has to allocate memory and then move for clustering and clusterWeights and sqDistances...if they sahre
    // pointer could be faster
    kmeansState.setClusters(clusterResults->clusters);
    kmeansState.resetClusterData(numClusters);

    pFinisher->finishClustering(&kmeansState);
    kmeansState.compareResults(clusterResults);

    return clusterResults;
}

template <typename precision, typename int_size>
std::shared_ptr<ClusterResults<precision, int_size>> CoresetKmeans<precision, int_size>::fit(
  const Matrix<precision, int_size>* const data, const int_size& numClusters, const int& numRestarts,
  const std::vector<precision>* const weights)
{
    throw std::runtime_error("Should not be calling this func.");
    return nullptr;
}
}  // namespace HPKmeans