#pragma once

#include "Definitions.hpp"

class Kmeans
{
private:
    /**
     * @brief Private member variables
     *
     */
    int numClusters;             // the number of clusters to cluster to data into
    int numRestarts;             // the number of times Kmeans should try to cluster the data
    float bestError;             // the error in the best clustering
    cluster_t clusters;          // the cluster centers
    cluster_t bestClusters;      // the best cluster centers
    clustering_t clustering;     // the cluster assignments for each data point
    clustering_t bestClustering; // the best cluster assignments

public:
    /**
     * @brief Construct a new Kmeans object.
     *
     * @param numClusters - The number of clusters.
     * @param numRestarts - The number of times to repeat the Kmeans calculation before returning an answer.
     */
    Kmeans(int numClusters, int numRestarts);

    /**
     * @brief Destroy the Kmeans object.
     *
     */
    ~Kmeans();

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
    cluster_t getClusters() { return bestClusters; }

    /**
     * @brief Get the bestError.
     *
     * @return int
     */
    float getError() { return bestError; };

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
};