#include <iostream>
#include <chrono>
#include <mpi.h>
#include "Kmeans.hpp"
#include "boost/random.hpp"
#include "boost/math/distributions/normal.hpp"
#include "time.h"
#include "Reader.hpp"
#include "Writer.hpp"
#include <string>
#include "SerialKMeans.hpp"
#include "OMPKmeans.hpp"
#include "MPIKmeans.hpp"

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

value_t *generateDataset_MPI(int numData, int numFeatures, int numClusters)
{
    value_t *data = new value_t[numData * numFeatures];
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

                    data[(j * numFeatures) + k] = normalDist0();
                }
                else if (i == 1)
                {
                    data[(j * numFeatures) + k] = normalDist10();
                }
                else if (i == 2)
                {
                    data[(j * numFeatures) + k] = normalDist100();
                }
                else if (i == 3)
                {
                    data[(j * numFeatures) + k] = normalDist50();
                }
                else
                {
                    data[(j * numFeatures) + k] = normalDist23();
                }
            }
        }
    }
    return data;
}

int main(int argc, char *argv[])
{
    int numData = 10000;
    int numFeatures = 2;
    int numClusters = 30;
    int numRestarts = 10;
    dataset_t data;

    bool MPI = true;
    if (MPI)
    {
        bool MPI_windows = false;

        // Runs MPI implementation of K++
        MPIKmeans kmeans(numClusters, numRestarts);
        MPI_Init(&argc, &argv);

        int rank, numProcs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

        // Run with scatter/gather MPI
        if (MPI_windows == false)
        {
            // value_t *data = generateDataset_MPI(numData, numFeatures, numClusters);
            CReader reader;
            reader.read("../test_10000_2.txt", 10000, 2);
            value_t *data = reader.getData();
            if (rank == 0)
            {
                auto start = std::chrono::high_resolution_clock::now();
                // kmeans.fit(numData, numFeatures, data, Kmeans::distanceL2);
                kmeans.fit(numData, numFeatures, data, numClusters, Kmeans::distanceL2);
                // std::cout << "HERE2" << std::endl;
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "Total time: " << duration.count() << std::endl;
                // std::cout << "Cluster Centers:" << std::endl;
                // for (auto &center : kmeans.getClusters())
                // {
                //     for (auto &coord : center)
                //     {
                //         std::cout << coord << " ";
                //     }
                //     std::cout << std::endl;
                // }
                DataSetWriter writer(kmeans.getClusters(), kmeans.getClustering());
                writer.writeClusters("../clusters_mpi_kpp.txt");
                writer.writeClustering("../clustering_mpi_kpp.txt");
            }
            else
            {
                kmeans.fit(numData, numFeatures, NULL, numClusters, Kmeans::distanceL2);
            }
        }
        // Run with MPI windows
        // else
        // {
        //     MPI_Win dataWin, clusteringWin, clusterCountWin, clusterCoordWin;
        //     value_t *data, clusterCoord;
        //     int *clustering, clusterCount;
        //     if (rank == 0)
        //     {
        //         // Shared Mem for data, clustering, cluster count, cluster coord
        //         MPI_Win_allocate_shared(numData * numFeatures * sizeof(value_t), sizeof(value_t), MPI_INFO_NULL, MPI_COMM_WORLD, &data, &dataWin);
        //         MPI_Win_allocate_shared(numData * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &clustering, &clusteringWin);
        //         MPI_Win_allocate_shared(numClusters * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &clusterCount, &clusterCountWin);
        //         MPI_Win_allocate_shared(numClusters * numFeatures * sizeof(value_t), sizeof(value_t), MPI_INFO_NULL, MPI_COMM_WORLD, &clusterCoord, &clusterCoordWin);
        //         data = generateDataset_MPI(numData, numFeatures, numClusters);
        //     }
        //     else
        //     {
        //         // Shared Mem for data, clustering, cluster count, cluster coord
        //         MPI_Win_allocate_shared(0, sizeof(value_t), MPI_INFO_NULL, MPI_COMM_WORLD, &data, &dataWin);
        //         MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &clustering, &clusteringWin);
        //         MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &clusterCount, &clusterCountWin);
        //         MPI_Win_allocate_shared(0, sizeof(value_t), MPI_INFO_NULL, MPI_COMM_WORLD, &clusterCoord, &clusterCoordWin);
        //     }

        //     MPI_Win_fence(MPI_MODE_NOPRECEDE, dataWin);
        //     MPI_Win_fence(MPI_MODE_NOPRECEDE, clusteringWin);
        //     MPI_Win_fence(MPI_MODE_NOPRECEDE, clusterCountWin);
        //     MPI_Win_fence(MPI_MODE_NOPRECEDE, clusterCoordWin);

        //     kmeans.setMPIWindows(dataWin, clusteringWin, clusterCountWin, clusterCoordWin);

        //     auto start = std::chrono::high_resolution_clock::now();
        //     kmeans.fit_MPI_win(numData, numFeatures, Kmeans::distanceL2);
        //     auto stop = std::chrono::high_resolution_clock::now();
        //     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        //     if (rank == 0)
        //     {
        //         std::cout << "Total time: " << duration.count() << std::endl;
        //     }

        //     MPI_Win_free(&dataWin);
        //     MPI_Win_free(&clusteringWin);
        //     MPI_Win_free(&clusterCountWin);
        //     MPI_Win_free(&clusterCoordWin);
        // }

        MPI_Finalize();
    }
    else
    {
        // cluster data
        // data = generateDataset(numData, numFeatures, numClusters);
        DataSetReader reader;
        reader.read("../test_10000_2.txt", 10000, 2);
        data = reader.getData();
        RNGType rng(time(NULL));
        boost::uniform_int<> intRange(0, data.size());
        boost::uniform_real<> floatRange(0, 1);
        boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
        boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

        // Kmeans kmeans(30, 30);
        // SerialKmeans kmeans(30, 30, intDistr, floatDistr);
        OMPKmeans kmeans(30, 30, intDistr, floatDistr);
        auto start = std::chrono::high_resolution_clock::now();
        // kmeans.fit(data, Kmeans::distanceL2); // kmeans++
        kmeans.fit(data, 30, Kmeans::distanceL2); // scalableKmeans
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Total time: " << duration.count() << std::endl;
        ClusterWriter writer(kmeans.getClusters(), kmeans.getClustering());
        // writer.writeClusters("../clusters_omp_kpp.txt");
        // writer.writeClustering("../clustering_omp_kpp.txt");
        writer.writeClusters("../clusters_omp_scale.txt");
        writer.writeClustering("../clustering_omp_scale.txt");

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
