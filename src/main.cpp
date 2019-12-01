#include <iostream>
#include <chrono>
#include <mpi.h>
#include "Kmeans.hpp"
#include "boost/random.hpp"
#include "boost/math/distributions/normal.hpp"
#include "time.h"

typedef boost::mt19937 RNGType;

dataset_t generateDataset(int numData, int numFeatures, int numClusters)
{
    // init normal distributions for generating psuedo data
    RNGType rng(time(NULL));
    boost::normal_distribution<> nd0(0.0, 1.0);
    boost::normal_distribution<> nd10(10, 2.0);
    boost::normal_distribution<> nd100(100, 10.0);
    boost::normal_distribution<> nd50(50, 5.0);
    boost::normal_distribution<> nd23(23, 7.0);
    boost::variate_generator<RNGType, boost::normal_distribution<>> normalDist0(rng, nd0);
    boost::variate_generator<RNGType, boost::normal_distribution<>> normalDist10(rng, nd10);
    boost::variate_generator<RNGType, boost::normal_distribution<>> normalDist100(rng, nd100);
    boost::variate_generator<RNGType, boost::normal_distribution<>> normalDist50(rng, nd50);
    boost::variate_generator<RNGType, boost::normal_distribution<>> normalDist23(rng, nd23);

    // generate data
    dataset_t data = dataset_t(numData, datapoint_t(numFeatures));
    for (int i = 0; i < numClusters; i++)
    {
        int min = i * numData / numClusters;
        int max = (i + 1) * numData / numClusters;
        for (int j = min; j < max; j++)
        {
            for (int k = 0; k < numFeatures; k++)
            {
                if (i == 0)
                {

                    data[j][k] = normalDist0();
                }
                else if (i == 1)
                {
                    data[j][k] = normalDist10();
                }
                else if (i == 2)
                {
                    data[j][k] = normalDist100();
                }
                else if (i == 3)
                {
                    data[j][k] = normalDist50();
                }
                else
                {
                    data[j][k] = normalDist23();
                }
            }
        }
    }
    return data;
}

void generateDataset(int numData, int numFeatures, int numClusters, value_t* data)
{
    // init normal distributions for generating psuedo data
    RNGType rng(time(NULL));
    boost::normal_distribution<> nd0(0.0, 1.0);
    boost::normal_distribution<> nd10(10, 2.0);
    boost::normal_distribution<> nd100(100, 10.0);
    boost::normal_distribution<> nd50(50, 5.0);
    boost::normal_distribution<> nd23(23, 7.0);
    boost::variate_generator<RNGType, boost::normal_distribution<>> normalDist0(rng, nd0);
    boost::variate_generator<RNGType, boost::normal_distribution<>> normalDist10(rng, nd10);
    boost::variate_generator<RNGType, boost::normal_distribution<>> normalDist100(rng, nd100);
    boost::variate_generator<RNGType, boost::normal_distribution<>> normalDist50(rng, nd50);
    boost::variate_generator<RNGType, boost::normal_distribution<>> normalDist23(rng, nd23);

    // generate data
    for (int i = 0; i < numClusters; i++)
    {
        int min = i * numData / numClusters;
        int max = (i + 1) * numData / numClusters;
        for (int j = min; j < max; j++)
        {
            for (int k = 0; k < numFeatures; k++)
            {
                if (i == 0)
                {

                    data[(j*numFeatures)+k] = normalDist0();
                }
                else if (i == 1)
                {
                    data[(j*numFeatures)+k] = normalDist10();
                }
                else if (i == 2)
                {
                    data[(j*numFeatures)+k] = normalDist100();
                }
                else if (i == 3)
                {
                    data[(j*numFeatures)+k] = normalDist50();
                }
                else
                {
                    data[(j*numFeatures)+k] = normalDist23();
                }
            }
        }
    }
}

int main(int argc, char* argv[])
{
    int numData = 1000;
    int numFeatures = 25;
    int numClusters = 5;
    int numRestarts = 10;
    dataset_t data;

    bool MPI = true;
    if (MPI)
    {
        // Runs MPI implementation of K++
        Kmeans kmeans(numClusters, 1);
        MPI_Init(&argc, &argv);

        int rank, numProcs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

        MPI_Win dataWin, clusteringWin, clusterCountWin, clusterCoordWin;
        value_t* data, clusterCoord;
        int* clustering, clusterCount;
        if (rank == 0)
        {
            // Shared Mem for data, clustering, cluster count, cluster coord
            MPI_Win_allocate_shared(numData * numFeatures * sizeof(value_t), sizeof(value_t), MPI_INFO_NULL, MPI_COMM_WORLD, &data, &dataWin);
            MPI_Win_allocate_shared(numData * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &clustering, &clusteringWin);
            MPI_Win_allocate_shared(numClusters * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &clusterCount, &clusterCountWin);
            MPI_Win_allocate_shared(numClusters * numFeatures * sizeof(value_t), sizeof(value_t), MPI_INFO_NULL, MPI_COMM_WORLD, &clusterCoord, &clusterCoordWin);
            generateDataset(numData, numFeatures, numClusters, data);
        }
        else
        {
            // Shared Mem for data, clustering, cluster count, cluster coord
            MPI_Win_allocate_shared(0, sizeof(value_t), MPI_INFO_NULL, MPI_COMM_WORLD, &data, &dataWin);
            MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &clustering, &clusteringWin);
            MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &clusterCount, &clusterCountWin);
            MPI_Win_allocate_shared(0, sizeof(value_t), MPI_INFO_NULL, MPI_COMM_WORLD, &clusterCoord, &clusterCoordWin);
        }
        MPI_Win_fence(MPI_MODE_NOPRECEDE, dataWin);
        MPI_Win_fence(MPI_MODE_NOPRECEDE, clusteringWin);
        MPI_Win_fence(MPI_MODE_NOPRECEDE, clusterCountWin);
        MPI_Win_fence(MPI_MODE_NOPRECEDE, clusterCoordWin);
        kmeans.setMPIWindows(dataWin, clusteringWin, clusterCountWin, clusterCoordWin);
        auto start = std::chrono::high_resolution_clock::now();
        kmeans.fit_MPI(numData, numFeatures, Kmeans::distanceL2);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Total time: " << duration.count() << std::endl;
        MPI_Win_free(&dataWin);
        MPI_Win_free(&clusteringWin);
        MPI_Win_free(&clusterCountWin);
        MPI_Win_free(&clusterCoordWin);
        MPI_Finalize();

    }
    else
    {
        // cluster data
        Kmeans kmeans(numClusters, 1);
        auto start = std::chrono::high_resolution_clock::now();
        // kmeans.fit(data, Kmeans::distanceL2); // kmeans++
        // kmeans.fit(data, numClusters / 3, Kmeans::distanceL2); // scalableKmeans
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Total time: " << duration.count() << std::endl;

        // std::cout << "Cluster Centers:" << std::endl;
        // for (auto &center : kmeans.getClusters())
        // {
        //     for (auto &coord : center.coords)
        //     {
        //         std::cout << coord << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // std::cout << "Clustering:" << std::endl;
        // for (auto &assignment : kmeans.getClustering())
        // {
        //     std::cout << assignment << " ";
        // }
    }
}
