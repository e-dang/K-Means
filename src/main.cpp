#include <iostream>
#include <chrono>
#include <mpi.h>
#include "Coresets.hpp"
#include "Reader.hpp"
#include "Writer.hpp"
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

    bool mpi = false;
    // cluster data
    Coresets kmeans(numClusters, 1, 4);
    // auto start = std::chrono::high_resolution_clock::now();
    // kmeans.fit(data, Coresets::distanceL2); // kmeans++
    // // kmeans.fit(data, numClusters / 3, Coresets::distanceL2); // scalableKmeans
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout << "Full Dataset Error: " << kmeans.getError() << std::endl;
    // std::cout << "Full Dataset Total time: " << duration.count() << std::endl;
    
    int coreset_size = 5000;

    if (mpi == false){
        // OMP coreset creation + fitting
        DataSetReader reader;
        reader.read("../test_10000_2.txt", 10000, 2);
        dataset_t data = reader.getData();
        auto start_coreset_creation = std::chrono::high_resolution_clock::now();
        kmeans.createCoreSet(data, coreset_size, Coresets::distanceL2);
        auto stop_coreset_creation = std::chrono::high_resolution_clock::now();
        auto duration_coreset_creation = std::chrono::duration_cast<std::chrono::microseconds>(stop_coreset_creation - start_coreset_creation);
        std::cout << "Coreset creation time omp: " << duration_coreset_creation.count() << std::endl;
        
        auto start_coreset_fitting = std::chrono::high_resolution_clock::now();
        kmeans.fit_coreset(Coresets::distanceL2);
        auto stop_coreset_fitting = std::chrono::high_resolution_clock::now();
        auto duration_coreset_fitting = std::chrono::duration_cast<std::chrono::microseconds>(stop_coreset_fitting - start_coreset_fitting);
        std::cout << "Coreset fitting error omp: " << kmeans.getError() << std::endl;
        std::cout << "Coreset fitting time omp: " << duration_coreset_fitting.count() << std::endl;
        ClusterWriter writer(kmeans.getClusters(), kmeans.getClustering());
        writer.writeClusters("../clusters_coreset_omp.txt");
        writer.writeClustering("../clustering_coreset_omp.txt");
    }
    else {
        // MPI Coreset creation + fitting
        MPI_Init(&argc, &argv);
        // value_t *data = generateDataset_MPI(numData, numFeatures, numClusters);
        CReader reader;
        reader.read("../test_10000_2.txt", 10000, 2);
        value_t *data = reader.getData();

        auto start_coreset_creation = std::chrono::high_resolution_clock::now();
        int rank, numProcs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        kmeans.createCoreSet_MPI(numData, numFeatures, data, coreset_size, Coresets::distanceL2);
        MPI_Finalize();
        auto stop_coreset_creation = std::chrono::high_resolution_clock::now();
        auto duration_coreset_creation = std::chrono::duration_cast<std::chrono::microseconds>(stop_coreset_creation - start_coreset_creation);
        std::cout << "Coreset creation time: " << duration_coreset_creation.count() << std::endl;

        if (rank == 0){
            auto start_coreset_fitting = std::chrono::high_resolution_clock::now();
            kmeans.fit_coreset(Coresets::distanceL2);
            auto stop_coreset_fitting = std::chrono::high_resolution_clock::now();
            auto duration_coreset_fitting = std::chrono::duration_cast<std::chrono::microseconds>(stop_coreset_fitting - start_coreset_fitting);
            std::cout << "Coreset fitting error: " << kmeans.getError() << std::endl;
            std::cout << "Coreset fitting time: " << duration_coreset_fitting.count() << std::endl;

            std::cout << "num clusters" << kmeans.getClusters().size() << std::endl;    
            ClusterWriter writer(kmeans.getClusters(), kmeans.getClustering());
            // writer.writeClusters("../clusters_omp_kpp.txt");
            // writer.writeClustering("../clustering_omp_kpp.txt");
            writer.writeClusters("../clusters_coreset_mpi.txt");
            writer.writeClustering("../clustering_coreset_mpi.txt");
        }
    }
}
