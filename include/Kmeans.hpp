#pragma once

#include "Definitions.hpp"
#include <mpi.h>

class Kmeans
{
private:
    /**
     * @brief Private member variables
     *
     */
    int numThreads;              // the number of threads to use
    int numClusters;             // the number of clusters to cluster to data into
    int numRestarts;             // the number of times Kmeans should try to cluster the data
    value_t bestError;           // the error in the best clustering
    clusters_t clusters;         // the cluster centers
    clusters_t bestClusters;     // the best cluster centers
    clustering_t clustering;     // the cluster assignments for each data point
    clustering_t bestClustering; // the best cluster assignments
    coreset_t coreset;           // the coreset to run clustering on if specified to do so
    MPI_Win dataWin;
    MPI_Win clusteringWin;
    MPI_Win clusterCoordWin;
    MPI_Win clusterCountWin;


    /**
     * @brief  Uses MPI window dataWin to retireve a chunk of the dataset
     * @note   
     * @param  start: Start datapoint index
     * @param  end: End datapoint inbex
     * @param  numFeatures: Number of features in each datapoint
     * @retval A dataset_t with the subset of the data
     */
    dataset_t getDataVecFromMPIWin(int start, int end, int numFeatures);

    /**
     * @brief  Uses MPI window clusterCoordWin to retrieve coordinates of a datapoint
     * @note   
     * @param  idx: Index of the datapoint
     * @param  numFeatures: Number of features in each datapoint
     * @retval A datapoint_t correspointing the index given
     */
    datapoint_t getClusterCoord(int idx, int numFeatures);

    /**
     * @brief  Uses MPI window clusterCountWin to retrieve the count for the specified cluster
     * @note   
     * @param  idx: Index of cluster
     * @retval count int
     */
    int getClusterCount(int idx);

    /**
     * @brief  Uses MPI window clusteringWin to retrieve cluster assigned to datapoint
     * @note   
     * @param  idx: Index of datapoint
     * @retval cluster number
     */
    int getClustering(int idx);

    /**
     * @brief  Sets datapoint coordinates in clusterCoordWin
     * @note   
     * @param  idx: index of cluster
     * @param  numFeatures: number of features in coordinates
     * @param  coord: datapoint_t corresponding to the coordinates to be set
     * @retval None
     */
    void setClusterCoord(int idx, int numFeatures, datapoint_t* coord);

    /**
     * @brief  Sets cluster count in clusterCountWin
     * @note   
     * @param  idx: index of cluster
     * @param  count: count to set
     * @retval None
     */
    void setClusterCount(int idx, int* count);

    
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
     * @brief Function for finding the closest cluster center to a datapoint and assigning that data point to that
     *        cluster. Uses MPI windows
     *
     * @param point - The datapoint to be considered.
     * @param pointIdx - The index of the datapoint in the dataset.
     * @param func - The distance function to use.
     * @return value_t - The square of the minimum distance.
     */
    value_t nearest_MPI(datapoint_t &point, int pointIdx, value_t (*func)(datapoint_t &, datapoint_t &), int clusterCount);

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

    /**
     * @brief An implementation of the Kmeans++ algorithm for initializing cluster centers. Does this by trying to
     *        maximize the distance between cluster centers. Uses MPI
     *
     * @param data - MPI shared data window.
     * @param func - The distance function to use.
     */
    void kPlusPlus_MPI(int numData, int numFeatures, value_t (*func)(datapoint_t &, datapoint_t &));

    /**
     * @brief Create the coreset used for representative kmeans clustering on the whole dataset. The coreset is stored
     *        in the private member variable coreset.
     *
     * @param data - The data to be clustered.
     * @param sampleSize - The number of datapoints to include in the coreset.
     * @param func - The distance function to use.
     */

    
    void createCoreSet(dataset_t &data, int &sampleSize, value_t (*func)(datapoint_t &, datapoint_t &));

public:
    /**
     * @brief Construct a new Kmeans object.
     *
     * @param numClusters - The number of clusters.
     * @param numRestarts - The number of times to repeat the Kmeans calculation before returning an answer.
     * @param numThreads - The number of threads to use. Defaults to 8.
     */
    Kmeans(int numClusters, int numRestarts, int numThreads = 8);

    /**
     * @brief Destroy the Kmeans object.
     *
     */
    ~Kmeans();

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
     * @brief Top level function that performs the clustering using lloyd's algorithm. Uses MPI implementation
     *
     * @param data - The data to be clustered.
     * @param func - The distance function to use.
     */
    void fit_MPI(int numData, int numFeatures, value_t (*func)(datapoint_t &, datapoint_t &));

    /**
     * @brief Function that performs the clustering on coresets using lloyd's algorithm with Kmeans ++ initialization.
     *
     * @param data - The data to be clustered.
     * @param func - The distance function to use.
     */
    void fit_coreset(value_t (*func)(datapoint_t &, datapoint_t &));

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
     * @brief Get the bestClustering object.
     *
     * @return clusteringPtr_t
     */
    clustering_t getClustering() { return bestClustering; }

    /**
     * @brief Get the bestClusters object.
     *
     * @return clustersPtr_t
     */
    clusters_t getClusters() { return bestClusters; }

    /**
     * @brief Get the bestError.
     *
     * @return int
     */
    value_t getError() { return bestError; };

    /**
     * @brief Get the numThreads object
     *
     * @return int
     */
    int getNumThreads() { return numThreads; };

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

    /**
     * @brief Set the numThreads object.
     *
     * @param numThreads
     * @return true
     * @return false
     */
    bool setNumThreads(int numThreads);

    /**
     * @brief The L2 norm between two data points.
     *
     * @param p1 - The first data point.
     * @param p2 - The second data point.
     * @return value_t - The distance.
     */
    static value_t distanceL2(datapoint_t &p1, datapoint_t &p2);

    void setMPIWindows(MPI_Win dataWin, MPI_Win clusteringWin, MPI_Win clusterCoordWin, MPI_Win clusterCountWin);
};