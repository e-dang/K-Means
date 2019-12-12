#include <iostream>
#include <chrono>
#include <mpi.h>
#include "Kmeans.hpp"
#include "Coresets.hpp"
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

int main(int argc, char *argv[])
{
    int numData = 10000;
    int numFeatures = 2;
    int numClusters = 30;
    int numRestarts = 10;
    int coresetSize = 5000;
    
    int trials = 1;
    int initIters = 4;

    float oversampling = numClusters / initIters;

    // Runs MPI implementation of Kmeans
    MPIKmeans kmeans(numClusters, numRestarts);
    Coresets coreset(numClusters, numRestarts);
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    CReader reader;
    reader.read("../test_10000_2.txt", numData, numFeatures);
    value_t *data = reader.getData();

    std::vector<int> kPPTimes;
    std::vector<int> scaleTimes;
    std::vector<int> coresetCreateTimes;
    std::vector<int> coresetFitTimes;
 

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

        // // Scalable kmeans
        // if (rank == 0)
        // {
        //     auto start = std::chrono::high_resolution_clock::now();
        //     kmeans.fit(numData, numFeatures, data, oversampling, Kmeans::distanceL2, initIters);
        //     auto stop = std::chrono::high_resolution_clock::now();
        //     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        //     std::cout << "Total time scale: " << duration.count() << std::endl;
        //     scaleTimes.push_back(duration.count());
        //     DataSetWriter writer(kmeans.getClusters(), kmeans.getClustering());
        //     writer.writeClusters("../clusters_mpi_scale.txt");
        //     writer.writeClustering("../clustering_mpi_scale.txt");
        // }
        // else
        // {
        //     kmeans.fit(numData, numFeatures, data, Kmeans::distanceL2);
        // }

        // mpi generate and fit coreset
        auto make_coreset_start = std::chrono::high_resolution_clock::now();
        coreset.createCoreSet_MPI(numData, numFeatures, data, coresetSize, Coresets::distanceL2);
        auto make_coreset_stop = std::chrono::high_resolution_clock::now();
        auto make_coreset_duration = std::chrono::duration_cast<std::chrono::microseconds>(make_coreset_stop - make_coreset_start);
        std::cout << "Total time coreset mpi: " << make_coreset_duration.count() << std::endl;
        coresetCreateTimes.push_back(make_coreset_duration.count());
        if (rank == 0)
        {
            auto start = std::chrono::high_resolution_clock::now();
            coreset.fit_coreset(Coresets::distanceL2);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "Total time coreset mpi: " << duration.count() << std::endl;
            coresetFitTimes.push_back(duration.count());
            ClusterWriter writer(coreset.getClusters(), coreset.getClustering());
            writer.writeClusters("../clusters_mpi_coresets.txt");
            writer.writeClustering("../clustering_mpi_coresets.txt");
        }
    }
    
    if(rank == 0)
    {
        std::ofstream out_kpp_mpi_file;
        std::string file_name = "../kpp_mpi_times-" + std::to_string(numData) + ".txt";
        out_kpp_mpi_file.open(file_name);

        for (auto time : kPPTimes)
        {
            out_kpp_mpi_file << time << " ";
        }
        out_kpp_mpi_file << "\n";
        out_kpp_mpi_file.close();

        // std::ofstream out_scale_file;
        // file_name = "../scale_times-" + std::to_string(numData) + ".txt";
        // out_scale_file.open(file_name);

        // for (auto time : scaleTimes)
        // {
        //     out_scale_file << time << " ";
        // }
        // out_scale_file << "\n";
        // out_scale_file.close();

        std::ofstream out_coreset_create_mpi_file;
        file_name = "../coreset_creation_mpi_times-" + std::to_string(numData) + ".txt";
        out_coreset_create_mpi_file.open(file_name);

        for (auto time : coresetCreateTimes)
        {
            out_coreset_create_mpi_file << time << " ";
        }
        out_coreset_create_mpi_file << "\n";
        out_coreset_create_mpi_file.close();

        std::ofstream out_coreset_fit_mpi_file;
        file_name = "../coreset_fit_mpi_times-" + std::to_string(numData) + ".txt";
        out_coreset_fit_mpi_file.open(file_name);

        for (auto time : coresetFitTimes)
        {
            out_coreset_fit_mpi_file << time << " ";
        }
        out_coreset_fit_mpi_file << "\n";
        out_coreset_fit_mpi_file.close();
    }

        MPI_Finalize();
}
