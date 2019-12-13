#include "SerialKMeans.hpp"
#include "OMPKmeans.hpp"
#include "Reader.hpp"
#include "Writer.hpp"
#include <chrono>
#include "Kmeans.hpp"
#include <vector>
// #include "omp.h"

int main(int argc, char *argv[])
{
    int numRestarts = 10;
    int numData = 100000;
    int numFeatures = 2;
    int numClusters = 30;
    int initIters = 4;
    float overSampling = numClusters / 2;
    int numThreads = 8;

    std::vector<double> times;

    DataSetReader reader;
    std::string inFile = "../test_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + ".txt";
    reader.read(inFile, numData, numFeatures);
    dataset_t data = reader.getData();

    RNGType rng(time(NULL));
    boost::uniform_int<> intRange(0, data.size());
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    // Serial Kmeans kPlusPlus
    SerialKmeans serialKmeanskpp(numClusters, numRestarts, intDistr, floatDistr);
    auto start = std::chrono::high_resolution_clock::now();
    serialKmeanskpp.fit(data, Kmeans::distanceL2);
    auto stop = std::chrono::high_resolution_clock::now();
    auto durationSerialKPP = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    times.push_back(durationSerialKPP.count());
    ClusterWriter writerKpp(serialKmeanskpp.getClusters(), serialKmeanskpp.getClustering());
    std::string serialKppClustersFile = "../clusters_serial_kpp_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + ".txt";
    std::string serialKppClusteringFile = "../clustering_serial_kpp_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + ".txt";
    writerKpp.writeClusters(serialKppClustersFile);
    writerKpp.writeClustering(serialKppClusteringFile);

    std::cout << "Serial KPlusPlus Done!" << std::endl;

    // Serial Kmeans Scaleable
    SerialKmeans serialKmeansScale(numClusters, numRestarts, intDistr, floatDistr);
    start = std::chrono::high_resolution_clock::now();
    serialKmeansScale.fit(data, overSampling, Kmeans::distanceL2, initIters);
    stop = std::chrono::high_resolution_clock::now();
    auto durationSerialScaleable = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    times.push_back(durationSerialScaleable.count());
    ClusterWriter writerScale(serialKmeansScale.getClusters(), serialKmeansScale.getClustering());
    std::string serialScaleClustersFile = "../clusters_serial_scale_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + ".txt";
    std::string serialScaleClusteringFile = "../clustering_serial_scale_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + ".txt";
    writerScale.writeClusters(serialScaleClustersFile);
    writerScale.writeClustering(serialScaleClusteringFile);

    std::cout << "Serial Scaleable Done!" << std::endl;

    // Omp Kmeans KPlusPlus
    // #pragma omp parallel
    //     {
    //         std::cout << omp_get_num_threads() << std::endl;
    //     }

    OMPKmeans ompKmeanskpp(numClusters, numRestarts, intDistr, floatDistr, numThreads);
    start = std::chrono::high_resolution_clock::now();
    ompKmeanskpp.fit(data, Kmeans::distanceL2);
    stop = std::chrono::high_resolution_clock::now();
    auto durationOmpKPP = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    times.push_back(durationOmpKPP.count());
    ClusterWriter writerOmpKpp(ompKmeanskpp.getClusters(), ompKmeanskpp.getClustering());
    std::string ompKppClustersFile = "../clusters_omp_kpp_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + "_" + std::to_string(numThreads) + ".txt";
    std::string ompKppClusteringFile = "../clustering_omp_kpp_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + "_" + std::to_string(numThreads) + ".txt";
    writerOmpKpp.writeClusters(ompKppClustersFile);
    writerOmpKpp.writeClustering(ompKppClusteringFile);

    std::cout << "OMP KPlusPlus Done!" << std::endl;

    // Omp Kmeans Scaleable
    OMPKmeans ompKmeansScale(numClusters, numRestarts, intDistr, floatDistr, numThreads);
    start = std::chrono::high_resolution_clock::now();
    ompKmeansScale.fit(data, overSampling, Kmeans::distanceL2, initIters);
    stop = std::chrono::high_resolution_clock::now();
    auto durationOmpScale = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    times.push_back(durationOmpScale.count());
    ClusterWriter writerOmpScale(ompKmeansScale.getClusters(), ompKmeansScale.getClustering());
    std::string ompScaleClustersFile = "../clusters_omp_scale_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + "_" + std::to_string(numThreads) + ".txt";
    std::string ompScaleClusteringFile = "../clustering_omp_scale_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + "_" + std::to_string(numThreads) + ".txt";
    writerOmpScale.writeClusters(ompScaleClustersFile);
    writerOmpScale.writeClustering(ompScaleClusteringFile);

    std::string timesFile = "../times_" + std::to_string(numData) + "_" + std::to_string(numFeatures) + "_serial.txt";
    writerScale.writeTimes(times, timesFile);

    // std::cout << "OMP Scaleable Done!" << std::endl;
    exit(0);
}