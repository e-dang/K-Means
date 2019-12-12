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
#include <fstream>

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
    int oversampling = 5;
    int trials = 1;

    // Runs MPI implementation of Kmeans
    MPIKmeans kmeans(numClusters, numRestarts);
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // value_t *data = generateDataset_MPI(numData, numFeatures, numClusters);
    CReader reader;
    reader.read("../test_10000_2.txt", 10000, 2);
    value_t *data = reader.getData();

    std::vector<float> kPPTimes;
    std::vector<float> scaleTimes;

    for(int j=0; j < trials; j++)
    {
        // K++
        if (rank == 0)
        {
            auto start = std::chrono::high_resolution_clock::now();
            kmeans.fit(numData, numFeatures, data, Kmeans::distanceL2);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "Total time kpp: " << duration.count() << std::endl;
            kPPTimes.push_back(duration.count());
            DataSetWriter writer(kmeans.getClusters(), kmeans.getClustering());
            writer.writeClusters("../clusters_mpi_kpp.txt");
            writer.writeClustering("../clustering_mpi_kpp.txt");
        }
        else
        {
            kmeans.fit(numData, numFeatures, data, Kmeans::distanceL2);
        }

        // Scalable kmeans
        if (rank == 0)
        {
            auto start = std::chrono::high_resolution_clock::now();
            kmeans.fit(numData, numFeatures, data, Kmeans::distanceL2);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "Total time scale: " << duration.count() << std::endl;
            scaleTimes.push_back(duration.count());
            DataSetWriter writer(kmeans.getClusters(), kmeans.getClustering());
            writer.writeClusters("../clusters_mpi_scale.txt");
            writer.writeClustering("../clustering_scale.txt");
        }
        else
        {
            kmeans.fit(numData, numFeatures, data, Kmeans::distanceL2);
        }
    }
    
    if(rank == 0)
    {
        std::ofstream out_kpp_file;
        out_kpp_file.open("../kpp_times.txt");

        for (auto time : kPPTimes)
        {
            out_kpp_file << time << " ";
        }
        out_kpp_file << "\n";
        out_kpp_file.close();

        std::ofstream out_scale_file;
        out_scale_file.open("../scale_times.txt");

        for (auto time : scaleTimes)
        {
            out_scale_file << time << " ";
        }
        out_scale_file << "\n";
        out_scale_file.close();
    }

        MPI_Finalize();
}
