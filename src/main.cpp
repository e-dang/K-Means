#include <chrono>
#include <iostream>

#include "Definitions.hpp"
#include "DistanceFunctors.hpp"
#include "KPlusPlus.hpp"
#include "Kmeans.hpp"
#include "Lloyd.hpp"
#include "Reader.hpp"
#include "Writer.hpp"
#include "mpi.h"

int main(int argc, char* argv[])
{
    int numRestarts = 5;
    int numData     = 100000;
    int numFeatures = 2;
    int numClusters = 30;

    int rank = 0, numProcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // VectorReader reader;
    MPIReader reader;
    std::string inFile = "../data/test_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + ".txt";
    reader.read(inFile, numData, numFeatures);
    // Matrix matrix(reader.getData(), numData, numFeatures);
    Matrix matrix(reader.getData(), numData / numProcs, numFeatures);

    // KPlusPlus kplusplus;
    // OptimizedKPlusPlus kplusplus;
    // OMPKPlusPlus kplusplus;
    // OMPOptimizedKPlusPlus kplusplus;
    MPIKPlusPlus kplusplus;
    // MPIOptimizedKPlusPlus kplusplus;
    // Lloyd lloyd;
    // OptimizedLloyd lloyd;
    // OMPLloyd lloyd;
    // OMPOptimizedLloyd lloyd;
    MPILloyd lloyd;
    // MPIOptimizedLloyd lloyd;
    EuclideanDistance distanceFunc;
    // Kmeans kmeans(&kplusplus, &lloyd, &distanceFunc);
    MPIKmeans kmeans(numData, &kplusplus, &lloyd, &distanceFunc);

    // std::cout << matrix.size() << " " << rank << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    kmeans.fit(&matrix, numClusters, numRestarts);
    auto stop              = std::chrono::high_resolution_clock::now();
    auto durationSerialKPP = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

    if (rank == 0)
    {
        ClusterDataWriter writer(kmeans.getClusterData(), numData, numFeatures);
        std::string serialKppClustersFile =
          "../data/clusters_serial_kpp_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + ".txt";
        std::string serialKppClusteringFile =
          "../data/clustering_serial_kpp_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + ".txt";
        writer.writeClusters(serialKppClustersFile);
        writer.writeClustering(serialKppClusteringFile);

        std::cout << "Serial KPlusPlus Done! Error: " << kmeans.getError() << " Time: " << durationSerialKPP.count()
                  << std::endl;
        auto clusterdata = kmeans.getClusterData();
        // std::cout << "Clustering" << std::endl;
        // int count = 0;
        // for (auto &val : clusterdata.mClustering)
        // {
        //     count++;
        //     std::cout << val << " ";
        // }

        std::cout << std::endl << std::endl;
        std::cout << "Clusters" << std::endl;
        clusterdata.mClusters.display();
    }
    MPI_Finalize();
}