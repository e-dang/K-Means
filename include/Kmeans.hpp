#pragma once

#include <memory>
#include <numeric>

#include "AbstractKmeansAlgorithms.hpp"
#include "CoresetCreator.hpp"
#include "DataClasses.hpp"
#include "Definitions.hpp"
#include "DistanceFunctors.hpp"

/**
 * @brief Abstract class that defines the interface for using a Kmeans class, which wraps an initialization and
 *        maximization algorithm together, along with a distance metric functor in order to cluster data. In addition
 *        this class also defines the member variables, setters, getters, and helper functions that each Kmeans
 *        concretion will need to function.
 */
class AbstractKmeans
{
protected:
    // Member variables
    std::unique_ptr<AbstractKmeansInitializer> pInitializer;
    std::unique_ptr<AbstractKmeansMaximizer> pMaximizer;
    std::shared_ptr<IDistanceFunctor> pDistanceFunc;

public:
    AbstractKmeans(IDistanceFunctor* distanceFunc) : pDistanceFunc(distanceFunc) {}

    /**
     * @brief Constructor for AbstractKmeans.
     *
     * @param initializer - A pointer to a class implementing a Kmeans initialization algorithm such as K++.
     * @param maximizer - A pointer to a class implementing a Kmeans maximization algorithm such as lloyd's algorithm.
     * @param distanceFunc - A pointer to a functor class used to calculate the distance between points, such as the
     *                       euclidean distance.
     */
    AbstractKmeans(AbstractKmeansInitializer* initializer, AbstractKmeansMaximizer* maximizer,
                   IDistanceFunctor* distanceFunc) :
        pInitializer(initializer), pMaximizer(maximizer), pDistanceFunc(distanceFunc)
    {
    }

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
    virtual ClusterResults fit(Matrix* data, const int& numClusters, const int& numRestarts);

    /**
     * @brief Interface for the top level function that initiates the clustering process.
     *
     * @param data - The data to be clustered.
     * @param numClusters - The number of clusters to cluster the data into.
     * @param numRestarts - The number of times to repeat the clustering process.
     * @param weights - The weights for each datapoint in the matrix.
     */
    virtual ClusterResults fit(Matrix* data, const int& numClusters, const int& numRestarts,
                               std::vector<value_t>* weights);

    // /**
    //  * @brief Set the initializer member variable.
    //  *
    //  * @param initializer - A pointer to an implementation of the AbstractKmeansInitializer class.
    //  */
    // void setInitializer(AbstractKmeansInitializer* initializer) { pInitializer = initializer; }

    // /**
    //  * @brief Set the maximizer member variable.
    //  *
    //  * @param maximizer - A pointer to an implementation of the AbstractKmeansMaximizer class.
    //  */
    // void setMaximizer(AbstractKmeansMaximizer* maximizer) { pMaximizer = maximizer; }

    /**
     * @brief Set the distanceFunc member variable.
     *
     * @param distanceFunc - A pointer to an implementation of the IDistanceFunctor class.
     */
    void setDistanceFunc(IDistanceFunctor* distanceFunc) { pDistanceFunc.reset(distanceFunc); }

protected:
    ClusterResults run(Matrix* data, const int& numClusters, const int& numRestarts, KmeansData* kmeansData);

    /**
     * @brief Helper function that takes in the resulting clusterData and squared distances of each datapoint to their
     *        assigned cluster and calculates the error. If the error is less than the previous run's error, the
     *        clusterData from the current run is stored in finalClusterData.
     *
     * @param clusterData - The clusterData from the current run.
     * @param sqDistances - The square distances from each to point to their assigned cluster.
     */
    void compareResults(ClusterData* clusterData, std::vector<value_t>* sqDistances, ClusterResults* clusterResults)
    {
        value_t currError = std::accumulate(sqDistances->begin(), sqDistances->end(), 0.0);

        if (clusterResults->mError > currError || clusterResults->mError < 0)
        {
            clusterResults->mError       = currError;
            clusterResults->mClusterData = *clusterData;
            clusterResults->mSqDistances = *sqDistances;
        }
    }

    virtual KmeansData initKmeansData(Matrix* data, std::vector<value_t>* weights)
    {
        return KmeansData(data, weights, pDistanceFunc, 0, data->getNumData(), std::vector<int>(1, data->getNumData()),
                          std::vector<int>(1, 0));
    }
};

/**
 * @brief Implementation of AbstractKmeans that can take serialized or threaded implementations of KmeansAlgorithms.
 */
class SharedMemoryKmeans : public AbstractKmeans
{
public:
    /**
     * @brief Construct a new Kmeans object. Calls base class' constructor.
     *
     * @param initializer - A pointer to a class implementing a Kmeans initialization algorithm such as K++.
     * @param maximizer - A pointer to a class implementing a Kmeans maximization algorithm such as lloyd's algorithm.
     * @param distanceFunc - A pointer to a functor class used to calculate the distance between points, such as the
     *                       euclidean distance.
     */
    SharedMemoryKmeans(AbstractKmeansInitializer* initializer, AbstractKmeansMaximizer* maximizer,
                       IDistanceFunctor* distanceFunc) :
        AbstractKmeans(initializer, maximizer, distanceFunc)
    {
    }

    /**
     * @brief Destroy the Kmeans object.
     */
    virtual ~SharedMemoryKmeans(){};
};

class MPIKmeans : public AbstractKmeans
{
private:
    int mTotalNumData;

public:
    /**
     * @brief Construct a new MPIKmeans object. Calls base class' constructor.
     *
     * @param initializer - A pointer to a class implementing a MPIKmeans initialization algorithm such as K++.
     * @param maximizer - A pointer to a class implementing a MPIKmeans maximization algorithm such as lloyd's
     * algorithm.
     * @param distanceFunc - A pointer to a functor class used to calculate the distance between points, such as the
     *                       euclidean distance.
     */
    MPIKmeans(const int& totalNumData, AbstractKmeansInitializer* initializer, AbstractKmeansMaximizer* maximizer,
              IDistanceFunctor* distanceFunc) :
        mTotalNumData(totalNumData), AbstractKmeans(initializer, maximizer, distanceFunc)
    {
    }

    /**
     * @brief Destroy the MPIKmeans object.
     */
    virtual ~MPIKmeans(){};

protected:
    KmeansData initKmeansData(Matrix* data, std::vector<value_t>* weights) override;
};

class CoresetKmeans : public AbstractKmeans
{
private:
    int mTotalNumData;
    int mSampleSize;
    std::unique_ptr<AbstractKmeans> pKmeans;
    std::unique_ptr<AbstractCoresetCreator> pCreator;
    std::unique_ptr<IClosestClusterFinder> pFinder;

public:
    CoresetKmeans(const int& totalNumData, const int& sampleSize, AbstractKmeans* kmeans,
                  AbstractCoresetCreator* creator, IClosestClusterFinder* finder, IDistanceFunctor* distanceFunc) :
        mTotalNumData(totalNumData),
        mSampleSize(sampleSize),
        pKmeans(kmeans),
        pCreator(creator),
        pFinder(finder),
        AbstractKmeans(distanceFunc)
    {
    }

    ClusterResults fit(Matrix* data, const int& numClusters, const int& numRestarts) override;

private:
    ClusterResults fit(Matrix* data, const int& numClusters, const int& numRestarts,
                       std::vector<value_t>* weights) override;
};