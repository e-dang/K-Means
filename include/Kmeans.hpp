#pragma once

#include <memory>
#include <numeric>

#include "CoresetClusteringFinisher.hpp"
#include "CoresetCreator.hpp"
#include "DataClasses.hpp"
#include "Definitions.hpp"
#include "DistanceFunctors.hpp"
#include "KmeansAlgorithms.hpp"
#include "KmeansDataCreator.hpp"

/**
 * @brief Abstract class that defines the interface for using a Kmeans class, which wraps an initialization and
 *        maximization algorithm together, along with a distance metric functor in order to cluster data. In addition
 *        this class also defines the member variables, setters, getters, and helper functions that each Kmeans
 *        concretion will need to function.
 */
class AbstractKmeans
{
protected:
    std::unique_ptr<AbstractKmeansInitializer> pInitializer;
    std::unique_ptr<AbstractKmeansMaximizer> pMaximizer;
    std::unique_ptr<IKmeansDataCreator> pDataCreator;
    std::shared_ptr<IDistanceFunctor> pDistanceFunc;

public:
    AbstractKmeans(IKmeansDataCreator* dataCreator, std::shared_ptr<IDistanceFunctor> distanceFunc) :
        pDataCreator(dataCreator), pDistanceFunc(distanceFunc){};

    /**
     * @brief Constructor for AbstractKmeans.
     *
     * @param initializer - A pointer to a class implementing a Kmeans initialization algorithm such as K++.
     * @param maximizer - A pointer to a class implementing a Kmeans maximization algorithm such as lloyd's algorithm.
     * @param distanceFunc - A pointer to a functor class used to calculate the distance between points, such as the
     *                       euclidean distance.
     */
    AbstractKmeans(AbstractKmeansInitializer* initializer, AbstractKmeansMaximizer* maximizer,
                   IKmeansDataCreator* dataCreator, std::shared_ptr<IDistanceFunctor> distanceFunc) :
        pInitializer(initializer), pMaximizer(maximizer), pDataCreator(dataCreator), pDistanceFunc(distanceFunc){};

    /**
     * @brief Destroy the Abstract Kmeans object.
     */
    virtual ~AbstractKmeans(){};

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
    virtual std::shared_ptr<ClusterResults> fit(Matrix* data, const int& numClusters, const int& numRestarts) = 0;

    /**
     * @brief Interface for the top level function that initiates the clustering process.
     *
     * @param data - The data to be clustered.
     * @param numClusters - The number of clusters to cluster the data into.
     * @param numRestarts - The number of times to repeat the clustering process.
     * @param weights - The weights for each datapoint in the matrix.
     */
    virtual std::shared_ptr<ClusterResults> fit(Matrix* data, const int& numClusters, const int& numRestarts,
                                                std::vector<value_t>* weights) = 0;

    /**
     * @brief Set the distanceFunc member variable.
     *
     * @param distanceFunc - A pointer to an implementation of the IDistanceFunctor class.
     */
    void setDistanceFunc(IDistanceFunctor* distanceFunc) { pDistanceFunc.reset(distanceFunc); }

protected:
    std::shared_ptr<ClusterResults> run(Matrix* data, const int& numClusters, const int& numRestarts,
                                        KmeansData* kmeansData);

    /**
     * @brief Helper function that takes in the resulting clusterData and squared distances of each datapoint to their
     *        assigned cluster and calculates the error. If the error is less than the previous run's error, the
     *        clusterData from the current run is stored in finalClusterData.
     *
     * @param clusterData - The clusterData from the current run.
     * @param sqDistances - The square distances from each to point to their assigned cluster.
     */
    void compareResults(ClusterData* clusterData, std::vector<value_t>* sqDistances,
                        std::shared_ptr<ClusterResults> clusterResults)
    {
        value_t currError = std::accumulate(sqDistances->begin(), sqDistances->end(), 0.0);

        if (clusterResults->mError > currError || clusterResults->mError < 0)
        {
            clusterResults->mError       = currError;
            clusterResults->mClusterData = *clusterData;
            clusterResults->mSqDistances = *sqDistances;
        }
    }
};

class WeightedKmeans : public AbstractKmeans
{
public:
    WeightedKmeans(AbstractKmeansInitializer* initializer, AbstractKmeansMaximizer* maximizer,
                   IKmeansDataCreator* dataCreator, std::shared_ptr<IDistanceFunctor> distanceFunc) :
        AbstractKmeans(initializer, maximizer, dataCreator, distanceFunc){};

    ~WeightedKmeans(){};

    std::shared_ptr<ClusterResults> fit(Matrix* data, const int& numClusters, const int& numRestarts) override;

    std::shared_ptr<ClusterResults> fit(Matrix* data, const int& numClusters, const int& numRestarts,
                                        std::vector<value_t>* weights) override;
};

class CoresetKmeans : public AbstractKmeans
{
private:
    int_fast32_t mSampleSize;
    std::unique_ptr<AbstractKmeans> pKmeans;
    std::unique_ptr<AbstractCoresetCreator> pCreator;
    std::unique_ptr<AbstractCoresetClusteringFinisher> pFinisher;

public:
    CoresetKmeans(const int_fast32_t& sampleSize, AbstractKmeans* kmeans, AbstractCoresetCreator* creator,
                  AbstractCoresetClusteringFinisher* finisher, IKmeansDataCreator* dataCreator,
                  std::shared_ptr<IDistanceFunctor> distanceFunc) :
        AbstractKmeans(dataCreator, distanceFunc),
        mSampleSize(sampleSize),
        pKmeans(kmeans),
        pCreator(creator),
        pFinisher(finisher){};

    ~CoresetKmeans(){};

    std::shared_ptr<ClusterResults> fit(Matrix* data, const int& numClusters, const int& numRestarts) override;

    std::shared_ptr<ClusterResults> fit(Matrix* data, const int& numClusters, const int& numRestarts,
                                        std::vector<value_t>* weights) override;
};