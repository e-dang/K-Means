#pragma once

#include "Definitions.hpp"

class Kmeans
{
private:
    /**
     * @brief Private member variables
     *
     */
    int numClusters;              // the number of clusters to cluster to data into
    int numRestarts;              // the number of times Kmeans should try to cluster the data
    float finalError;             // the error in the final clustering
    cluster_t clusters;           // the cluster centers
    cluster_t finalClusters;      // the final cluster centers
    clustering_t clustering;      // the cluster assignments for each data point
    clustering_t finalClustering; // the final cluster assignment

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
};