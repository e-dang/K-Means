#include "SerialKMeans.hpp"
#include "OMPKmeans.hpp"
#include "Reader.hpp"
#include "Writer.hpp"
#include <chrono>
#include "Kmeans.hpp"
#include <vector>

int main(int argc, char *argv[])
{
    int numRestarts = 10;
    int numData = 1000000;
    int numFeatures = 2;
    int numClusters = 10000;
    int initIters = 4;
    float overSampling = numClusters / initIters;

    std::vector<float> times;

    DataSetReader reader;
    reader.read("data_1000000_2.txt", numData, numFeatures);
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
    auto durationSerialKPP = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    times.push_back(durationSerialKPP.count());
    ClusterWriter writerKpp(serialKmeanskpp.getClusters(), serialKmeanskpp.getClustering());
    writerKpp.writeClusters("../clusters_serial_kpp.txt");
    writerKpp.writeClustering("../clustering_serial_kpp.txt");

    // Serial Kmeans Scaleable
    SerialKmeans serialKmeansScale(numClusters, numRestarts, intDistr, floatDistr);
    auto start = std::chrono::high_resolution_clock::now();
    serialKmeansScale.fit(data, overSampling, Kmeans::distanceL2, initIters);
    auto stop = std::chrono::high_resolution_clock::now();
    auto durationSerialScaleable = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    times.push_back(durationSerialScaleable.count());
    ClusterWriter writerScale(serialKmeansScale.getClusters(), serialKmeansScale.getClustering());
    writerScale.writeClusters("../clusters_serial_scale.txt");
    writerScale.writeClustering("../clustering_serial_scale.txt");

    // Omp Kmeans KPlusPlus
    OMPKmeans ompKmeanskpp(numClusters, numClusters, intDistr, floatDistr);
    auto start = std::chrono::high_resolution_clock::now();
    ompKmeanskpp.fit(data, Kmeans::distanceL2);
    auto stop = std::chrono::high_resolution_clock::now();
    auto durationOmpKPP = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    times.push_back(durationOmpKPP.count());
    ClusterWriter writerOmpKpp(ompKmeanskpp.getClusters(), ompKmeanskpp.getClustering());
    writerOmpKpp.writeClusters("../clusters_omp_kpp.txt");
    writerOmpKpp.writeClustering("../clustering_omp_kpp.txt");

    // Omp Kmeans Scaleable
    OMPKmeans ompKmeansScale(numClusters, numClusters, intDistr, floatDistr);
    auto start = std::chrono::high_resolution_clock::now();
    ompKmeansScale.fit(data, overSampling, Kmeans::distanceL2, initIters);
    auto stop = std::chrono::high_resolution_clock::now();
    auto durationOmpScale = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    times.push_back(durationOmpScale.count());
    ClusterWriter writerOmpScale(ompKmeansScale.getClusters(), ompKmeansScale.getClustering());
    writerOmpScale.writeClusters("../clusters_omp_scale.txt");
    writerOmpScale.writeClustering("../clustering_omp_scale.txt");

    writerOmpScale.writeTimes(times, "../times.txt");

    exit(0);
}