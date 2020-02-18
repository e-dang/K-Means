#pragma once

#include <memory>
#include <numeric>

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
#include "KmeansAlgorithms/CoresetCreator.hpp"
#include "KmeansAlgorithms/KmeansAlgorithms.hpp"
#include "Strategies/CoresetClusteringFinisher.hpp"
#include "Strategies/KmeansDataCreator.hpp"
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
template <typename precision = double, typename int_size = int32_t>
class AbstractKmeans
{
protected:
    std::unique_ptr<AbstractKmeansInitializer<precision, int_size>> pInitializer;
    std::unique_ptr<AbstractKmeansMaximizer<precision, int_size>> pMaximizer;
    std::unique_ptr<IKmeansDataCreator<precision, int_size>> pDataCreator;
    std::shared_ptr<IDistanceFunctor<precision>> pDistanceFunc;

public:
    AbstractKmeans(IKmeansDataCreator<precision, int_size>* dataCreator,
                   std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        pDataCreator(dataCreator), pDistanceFunc(distanceFunc)
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
                   IKmeansDataCreator<precision, int_size>* dataCreator,
                   std::shared_ptr<IDistanceFunctor<precision>> distanceFunc) :
        pInitializer(initializer), pMaximizer(maximizer), pDataCreator(dataCreator), pDistanceFunc(distanceFunc)
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
    void setDistanceFunc(IDistanceFunctor<precision>* distanceFunc) { pDistanceFunc.reset(distanceFunc); }

protected:
    std::shared_ptr<ClusterResults<precision, int_size>> run(const Matrix<precision, int_size>* const data,
                                                             const int_size& numClusters, const int& numRestarts,
                                                             KmeansData<precision, int_size>* const kmeansData);

    /**
     * @brief Helper function that takes in the resulting clusterData and squared distances of each datapoint to their
     *        assigned cluster and calculates the error. If the error is less than the previous run's error, the
     *        clusterData from the current run is stored in finalClusterData.
     *
     * @param clusterData - The clusterData from the current run.
     * @param sqDistances - The square distances from each to point to their assigned cluster.
     */
    void compareResults(ClusterData<precision, int_size>* const clusterData, std::vector<precision>* const sqDistances,
                        std::shared_ptr<ClusterResults<precision, int_size>> clusterResults);
};

template <typename precision = double, typename int_size = int32_t>
class WeightedKmeans : public AbstractKmeans<precision, int_size>
{
public:
    WeightedKmeans(AbstractKmeansInitializer<precision, int_size>* initializer,
                   AbstractKmeansMaximizer<precision, int_size>* maximizer,
                   IKmeansDataCreator<precision, int_size>* dataCreator,
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

template <typename precision = double, typename int_size = int32_t>
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
                  IKmeansDataCreator<precision, int_size>* dataCreator,
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
  KmeansData<precision, int_size>* const kmeansData)
{
    std::shared_ptr<ClusterResults<precision, int_size>> clusterResults =
      std::make_shared<ClusterResults<precision, int_size>>();

    pInitializer->setKmeansData(kmeansData);
    pMaximizer->setKmeansData(kmeansData);

    for (int i = 0; i < numRestarts; ++i)
    {
        std::vector<precision> distances(kmeansData->totalNumData, 1);
        ClusterData<precision, int_size> clusterData(kmeansData->totalNumData, data->cols(), numClusters);

        kmeansData->setClusterData(&clusterData);
        kmeansData->setSqDistances(&distances);

        pInitializer->initialize();
        pMaximizer->maximize();

        compareResults(&clusterData, &distances, clusterResults);
    }

    return clusterResults;
}

template <typename precision, typename int_size>
void AbstractKmeans<precision, int_size>::compareResults(
  ClusterData<precision, int_size>* const clusterData, std::vector<precision>* const sqDistances,
  std::shared_ptr<ClusterResults<precision, int_size>> clusterResults)
{
    precision currError = std::accumulate(sqDistances->begin(), sqDistances->end(), 0.0);

    if (clusterResults->error > currError || clusterResults->error < 0)
    {
        clusterResults->error       = currError;
        clusterResults->clusterData = std::move(*clusterData);
        clusterResults->sqDistances = std::move(*sqDistances);
    }
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
    auto kmeansData = this->pDataCreator->create(data, weights, this->pDistanceFunc);
    return this->run(data, numClusters, numRestarts, &kmeansData);
}

template <typename precision, typename int_size>
std::shared_ptr<ClusterResults<precision, int_size>> CoresetKmeans<precision, int_size>::fit(
  const Matrix<precision, int_size>* const data, const int_size& numClusters, const int& numRestarts)
{
    auto kmeansData = this->pDataCreator->create(data, nullptr, this->pDistanceFunc);

    auto coreset = pCreator->createCoreset(data);

    auto clusterResults = pKmeans->fit(&coreset.data, numClusters, numRestarts, &coreset.weights);
    clusterResults->clusterData.clustering.resize(kmeansData.totalNumData);
    clusterResults->sqDistances.resize(kmeansData.totalNumData);
    std::fill(clusterResults->clusterData.clustering.begin(), clusterResults->clusterData.clustering.end(), -1);
    std::fill(clusterResults->sqDistances.begin(), clusterResults->sqDistances.end(), -1);

    kmeansData.setClusterData(&clusterResults->clusterData);
    kmeansData.setSqDistances(&clusterResults->sqDistances);

    clusterResults->error = pFinisher->finishClustering(&kmeansData);

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