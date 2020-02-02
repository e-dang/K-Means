#include <chrono>
#include <iostream>

#include "Averager.hpp"
#include "CoresetCreator.hpp"
#include "Definitions.hpp"
#include "DistanceFunctors.hpp"
#include "KPlusPlus.hpp"
#include "Kmeans.hpp"
#include "Lloyd.hpp"
#include "RandomSelector.hpp"
#include "Reader.hpp"
#include "Writer.hpp"
#include "mpi.h"

int main(int argc, char* argv[])
{
    int numRestarts = 5;
    int numData     = 100000;
    int numFeatures = 2;
    int numClusters = 30;
    int sampleSize  = 1000;

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
    // MPIKPlusPlus kplusplus;
    // MPIOptimizedKPlusPlus kplusplus;
    // Lloyd lloyd;
    // OptimizedLloyd lloyd;
    // OMPLloyd lloyd;
    // OMPOptimizedLloyd lloyd;
    // MPILloyd lloyd;
    // MPIOptimizedLloyd lloyd;
    // EuclideanDistance distanceFunc;
    CoresetKmeans kmeans(
      numData, sampleSize,
      new MPIKmeans(sampleSize, new MPIOptimizedKPlusPlus, new MPIOptimizedLloyd, new EuclideanDistance),
      new MPICoresetCreator(numData, sampleSize, new MultiWeightedRandomSelector, new VectorAverager,
                            new EuclideanDistance),
      new ClosestClusterFinder(nullptr), new EuclideanDistance);
    // CoresetKmeans kmeans(sampleSize, new Kmeans(&kplusplus, &lloyd, &distanceFunc),
    //                      new OMPCoresetCreator(new MultiWeightedRandomSelector, new OMPVectorAverager),
    //                      new ClosestClusterFinder(nullptr), &distanceFunc);

    // Kmeans kmeans(&kplusplus, &lloyd, &distanceFunc);
    // MPIKmeans kmeans(numData, &kplusplus, &lloyd, &distanceFunc);

    // std::cout << matrix.size() << " " << rank << std::endl;
    auto start             = std::chrono::high_resolution_clock::now();
    auto clusterResults    = kmeans.fit(&matrix, numClusters, numRestarts);
    auto stop              = std::chrono::high_resolution_clock::now();
    auto durationSerialKPP = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

    if (rank == 0)
    {
        ClusterDataWriter writer(clusterResults.mClusterData, numData, numFeatures);
        std::string serialKppClustersFile =
          "../data/clusters_serial_kpp_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + ".txt";
        std::string serialKppClusteringFile =
          "../data/clustering_serial_kpp_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + ".txt";
        writer.writeClusters(serialKppClustersFile);
        writer.writeClustering(serialKppClusteringFile);

        std::cout << "Serial KPlusPlus Done! Error: " << clusterResults.mError << " Time: " << durationSerialKPP.count()
                  << std::endl;
        auto clusterdata = clusterResults.mClusterData;
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