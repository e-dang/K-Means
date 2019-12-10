#pragma once
#include "Definitions.hpp"

class SerialKmeans
{
private:
    int numClusters;
    int numRestarts;
    int numThreads;

    double bestError;
    clusters_t clusters;
    clusters_t bestClusters;
    clustering_t clustering;
    clustering_t bestClustering;

public:
    /**
     * @brief Construct a new Kmeans object.
     *
     * @param numClusters - The number of clusters.
     * @param numRestarts - The number of times to repeat the Kmeans calculation before returning an answer.
     */
    SerialKmeans(int numClusters, int numRestarts);

    /**
     * @brief Destroy the Kmeans object.
     *
     */
    ~SerialKmeans();

    /**
     * @brief Top level function that performs the clustering using lloyd's algorithm.
     *
     * @param data - The data to be clustered.
     * @param func - The distance function to use.
     */
    void fit(dataset_t &data, value_t (*func)(datapoint_t &, datapoint_t &));

    /**
     * @brief Overloaded variant that uses Keamns Parallel as an initialization method and a quick version of the lloyd
     *        algorithm.
     *
     * @param data - The data to be clustered.
     * @param overSampling - The over sampling factor to be used in initialization.
     * @param func - The distance function to use.
     * @param initIters - The number of iters to do in initialization.
     */
    void fit(dataset_t &data, int overSampling, value_t (*func)(datapoint_t &, datapoint_t &), int initIters = 3);

    /**
     * @brief Get the numClusters object.
     *
     * @return int
     */
    int getNumClusters() { return numClusters; }

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
    clusters_t getClusters() { return bestClusters; }

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
     * @brief An implementation of the Kmeans++ algorithm for initializing cluster centers. Does this by trying to
     *        maximize the distance between cluster centers.
     *
     * @param data - The data that is being clustered.
     * @param func - The distance function to use.
     */
    void kPlusPlus(dataset_t &data, value_t (*func)(datapoint_t &, datapoint_t &));

    /**
     * @brief An implementation of the Kmeans Parallel initialization algorithm.
     *
     * @param data - The data that is being clustered
     * @param overSampling - The expected amount of clusters to sample in each iteration
     * @param func - The distance function to use
     * @param initIters - The number of iterations of cluster sampling to do
     * @return std::vector<value_t>
     */
    std::vector<value_t> scaleableKmeans(dataset_t &data, int &overSampling,
                                         value_t (*func)(datapoint_t &, datapoint_t &), int initIters = 3);

    /**
     * @brief Function for finding the closest cluster center to a datapoint and assigning that data point to that
     *        cluster.
     *
     * @param point - The datapoint to be considered.
     * @param pointIdx - The index of the datapoint in the dataset.
     * @param func - The distance function to use.
     * @return value_t - The square of the minimum distance.
     */
    value_t nearest(datapoint_t &point, int &pointIdx, value_t (*func)(datapoint_t &, datapoint_t &));

    /**
     * @brief - Finds the closest cluster to a datapoint out of the set of newly added clusters and the cluster that the
     *          datapoint was already assigned to. If the datapoint point is closer to a new cluster than its old
     *          cluster, the function update the distance vector and the cluster assignment for that point. This
     *          function is only used in the initialization step.
     *
     * @param point - The datapoint to be considered.
     * @param pointIdx - The index of the datapoint in the dataset.
     * @param clusterIdx - The starting index of the set of new clusters.
     * @param distances - The distance vector.
     * @param func - The distance function to use.
     */
    void smartClusterUpdate(datapoint_t &point, int &pointIdx, int &clusterIdx, std::vector<value_t> &distances,
                            value_t (*func)(datapoint_t &, datapoint_t &));
};