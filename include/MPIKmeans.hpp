#pragma once

#include "Definitions.hpp"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"

typedef boost::mt19937 RNGType;

class MPIKmeans
{
private:
    /**
     * @brief Private member variables
     *
     */
    int numThreads;  // the number of threads to use
    int numClusters; // the number of clusters to cluster to data into
    int numRestarts; // the number of times Kmeans should try to cluster the data

    double bestError;
    clustering_t bestClustering; // the best cluster assignments
    std::vector<int> bestClusterCount;
    dataset_t bestClusterCoord;

    int startIdx_MPI;
    int endIdx_MPI;
    dataset_t data_MPI;
    clustering_t clustering_MPI;
    clustering_t clusteringChunk_MPI;
    std::vector<int> clusterCount_MPI;
    dataset_t clusterCoord_MPI;
    std::vector<int> vDisps_MPI;
    std::vector<int> vLens_MPI;

public:
    MPIKmeans(int numClusters, int numRestarts);
    ~MPIKmeans();

    void fit(int numData, int numFeatures, value_t *data, value_t (*func)(datapoint_t &, datapoint_t &));

    void fit(int numData, int numFeatures, value_t *data, int overSampling,
             value_t (*func)(datapoint_t &, datapoint_t &), int initIters = 5);

    /**
     * @brief Get the numClusters object.
     *
     * @return int
     */
    int getNumClusters()
    {
        return numClusters;
    }

    /**
     * @brief Get the numRestarts object.
     *
     * @return int
     */
    int getNumRestarts() { return numRestarts; }

    /**
     * @brief Get the bestClusters object.
     *
     * @return clusters_t
     */
    dataset_t getClusters() { return bestClusterCoord; }

    /**
     * @brief Get the bestClustering object.
     *
     * @return clustering_t
     */
    clustering_t getClustering() { return bestClustering; }

    /**
     * @brief Get the bestError.
     *
     * @return int
     */
    double getError() { return bestError; };

    /**
     * @brief Set the numClusters object.
     *
     * @param numClusters
     * @return true
     * @return false
     */
    bool setNumClusters(int numClusters);

    /**
     * @brief Set the numRestarts object.
     *
     * @param numRestarts
     * @return true
     * @return false
     */
    bool setNumRestarts(int numRestarts);

private:
    /**
     * @brief  Initializes all the member variables used for scatter/gather MPI
     * @note
     * @param  numData: Number of datapoints
     * @param  numFeatures: Features per datapoint
     * @param  data: c-array with all data
     * @retval None
     */
    void initMPIMembers(int numData, int numFeatures, value_t *data = NULL);

    void kPlusPlus(int numData, int numFeatures, value_t *data, value_t (*func)(datapoint_t &, datapoint_t &),
                   boost::variate_generator<RNGType, boost::uniform_int<>> intDistr,
                   boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr);

    /**
     * @brief An implementation of the Kmeans Parallel initialization algorithm using MPI.
     *
     * @param data - The data that is being clustered
     * @param overSampling - The expected amount of clusters to sample in each iteration
     * @param func - The distance function to use
     * @param initIters - The number of iterations of cluster sampling to do
     * @return std::vector<value_t>
     */
    std::vector<value_t> scaleableKmeans(int &numData, int &numFeatures, value_t *data, int &overSampling,
                                         value_t (*func)(datapoint_t &, datapoint_t &),
                                         boost::variate_generator<RNGType, boost::uniform_int<>> intDistr,
                                         boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr,
                                         int initIters);

    value_t nearest(datapoint_t &point, int pointIdx, value_t (*func)(datapoint_t &, datapoint_t &), int clusterCount);

    void smartClusterUpdate(datapoint_t &point, int &pointIdx, int &prevNumClusters, int &clusterCount, value_t *distances,
                            dataset_t &localCoords, value_t (*func)(datapoint_t &, datapoint_t &));

    /**
     * @brief  Converts a c-array of datapoints into the dataset_t object
     * @note
     * @param  data: Pointer to the data array
     * @param  size: number of datapoints
     * @param  numFeatures: Number of features per datapoing
     * @retval
     */
    dataset_t arrayToDataset(value_t *data, int size, int numFeatures);

    void bcastClusters(int &clusterCount);
};