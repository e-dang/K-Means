#include "Kmeans.hpp"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <assert.h>

typedef boost::mt19937 RNGType;

Kmeans::Kmeans(int numClusters, int numRestarts, int numThreads) : numClusters(numClusters), numRestarts(numRestarts),
                                                                   numThreads(numThreads)
{
    
    bestError = INT_MAX;
    setNumThreads(numThreads);
}

Kmeans::~Kmeans()
{
}

void Kmeans::fit(dataset_t &data, value_t (*func)(datapoint_t &, datapoint_t &))
{

    int changed;
    int numData = data.size();
    int numFeatures = data[0].size();
    value_t currError;

    for (int run = 0; run < numRestarts; run++)
    {
        bestClusters = clusters_t();
        bestClustering = clustering_t(numData, -1);

        // initialize clusters with k++ algorithm
        kPlusPlus(data, func);

        do
        {
            // reinitialize clusters
            for (int i = 0; i < numClusters; i++)
            {
                clusters[i] = {0, datapoint_t(numFeatures, 0.)};
            }

            // calc sum of each feature for all points belonging to a cluster
            for (int i = 0; i < numData; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    clusters[clustering[i]].coords[j] += data[i][j];
                }
                clusters[clustering[i]].count++;
            }

            // divide sum by number of points belonging to the cluster to obtain average
            for (int i = 0; i < numClusters; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    clusters[i].coords[j] /= clusters[i].count;
                }
            }

            // reassign points to cluster
            changed = 0;
#pragma omp parallel for shared(data), schedule(static), reduction(+ \
                                                                   : changed)
            for (int i = 0; i < numData; i++)
            {
                int before = clustering[i];
                nearest(data[i], i, func);
                if (before != clustering[i])
                {
                    changed++;
                }
            }
        } while (changed > (numData >> 10)); // do until 99.9% of data doesnt change

        // get total sum of distances from each point to their cluster center
        currError = 0;
        for (int i = 0; i < numData; i++)
        {
            currError += std::pow(func(data[i], clusters[clustering[i]].coords), 2);
        }

        // if this round produced lowest error, keep clustering
        if (currError < bestError)
        {
            bestError = currError;
            bestClustering = clustering;
            bestClusters = clusters;
        }
    }
}

void Kmeans::fit_MPI(int numData, int numFeatures, value_t (*func)(datapoint_t &, datapoint_t &))
{
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int changed;
    value_t currError;

    if(rank == 0)
    {
        clusters = clusters_t(numClusters);
        clustering = clustering_t(numData);
        // clustering = clustering_t(numData, -1);
    }

    for (int run = 0; run < numRestarts; run++)
    {
        // initialize clusters with k++ algorithm
        kPlusPlus_MPI(numData, numFeatures, func);

        do
        {
            if(rank == 0)
            {
                // reinitialize clusters
                for (int i = 0; i < numClusters; i++)
                {
                    int count = 0;
                    setClusterCount(i, &count);
                    datapoint_t tempCoord(numFeatures, 0.);
                    setClusterCoord(i, numFeatures, &tempCoord);
                    // clusters[i] = {0, datapoint_t(numFeatures, 0.)};
                }
            

                // calc sum of each feature for all points belonging to a cluster
                for (int i = 0; i < numData; i++)
                {
                    int clst = getClustering(i);
                    assert(clst < numClusters);
                    datapoint_t coord = getClusterCoord(clst, numFeatures);
                    datapoint_t data = getDataVecFromMPIWin(i, i+1, numFeatures)[0];
                    for (int j = 0; j < numFeatures; j++)
                    {
                        coord[j] += data[j];
                        // clusters[clustering[i]].coords[j] +=  data[i][j];
                    }
                    setClusterCoord(clst, numFeatures, &coord);
                    int count = getClusterCount(i);
                    count++;
                    setClusterCount(clst, &count);
                    // clusters[clustering[i]].count++;
                }

                // divide sum by number of points belonging to the cluster to obtain average
                for (int i = 0; i < numClusters; i++)
                {
                    datapoint_t coord = getClusterCoord(i, numFeatures);
                    int count = getClusterCount(i);
                    for (int j = 0; j < numFeatures; j++)
                    {
                        coord[j] /= count;
                        // clusters[i].coords[j] /= clusters[i].count;
                    }
                    setClusterCoord(i, numFeatures, &coord);
                }
            }
            MPI_Win_fence(MPI_MODE_NOPRECEDE, clusterCountWin);
            MPI_Win_fence(MPI_MODE_NOPRECEDE, clusterCoordWin);

            // reassign points to cluster
            changed = 0;
            int local_changed = 0;

            // Datapoints allocated for each process to compute
            int chunk = numData / numProcs;
            int scrap = chunk + (numData % numProcs);
            // Start index for data
            int start = chunk * rank;
            // End index of Data
            int end = start + chunk - 1;
            // Last process gets leftover datapoints
            if (rank == (numProcs - 1)) end = start + scrap - 1;
            dataset_t data = getDataVecFromMPIWin(start, end, numFeatures);
            
            for (int i = 0; i < data.size(); i++)
            {
                int before, current;
                MPI_Get(&before, 1, MPI_INT, 0, i+start, 1, MPI_INT, clusteringWin);
                nearest_MPI(data[i], i+start, func, numClusters);
                MPI_Get(&current, 1, MPI_INT, 0, i+start, 1, MPI_INT, clusteringWin);
                if (before != current)
                {
                    local_changed++;
                }
            }
            // Aggregate changed
            MPI_Allreduce(&local_changed, &changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        } while (changed > (numData >> 10)); // do until 99.9% of data doesnt change



        // get total sum of distances from each point to their cluster center
        currError = 0;
        value_t localError = 0;

        // Datapoints allocated for each process to compute
        int chunk = numData / numProcs;
        int scrap = chunk + (numData % numProcs);
        // Start index for data
        int start = chunk * rank;
        // End index of Data
        int end = start + chunk - 1;
        // Last process gets leftover datapoints
        if (rank == (numProcs - 1)) end = start + scrap - 1;
        dataset_t data = getDataVecFromMPIWin(start, end, numFeatures);

        for (int i = 0; i < data.size(); i++)
        {
            int idx = i + start;
            int clst = getClustering(idx);
            assert(clst < numClusters);
            datapoint_t coord = getClusterCoord(clst, numFeatures);
            localError += std::pow(func(data[i], coord), 2);
            // localError += std::pow(func(data[i], clusters[clustering[i]].coords), 2);
        }

        MPI_Reduce(&localError, &currError, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if(rank == 0)
        {
            // if this round produced lowest error, keep clustering
            if (currError < bestError)
            {
                bestError = currError;

                // Fill best clustering from shared window
                MPI_Get(&bestClustering[0], 1, MPI_INT, 0, numData, 1, MPI_INT, clusteringWin);
                
                // Fill best clusters from shared window
                for(int i = 0; i < numClusters; i++)
                {
                    bestClusters[i].count = getClusterCount(i);
                    bestClusters[i].coords = getClusterCoord(i, numFeatures);
                }
                
            }
        }
    }
}

void Kmeans::kPlusPlus_MPI(int numData, int numFeatures, value_t (*func)(datapoint_t &, datapoint_t &))
{
    int rank, numProcs, clusterCount = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);


    RNGType rng(time(NULL));
    boost::uniform_int<> intRange(0, numData);
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    value_t sum = 0;
    value_t distances[numData];

    if(rank == 0)
    {
        int ranNum = intDistr();
        dataset_t randomDataPoint = getDataVecFromMPIWin(ranNum, ranNum+1, numFeatures);
        // initialize first cluster randomly
        setClusterCoord(0, numFeatures, &randomDataPoint[0]);
        clusterCount++;
        // clusters.push_back({0, datapoint_t(randomDataPoint[0])});
    }

    for (int clustIdx = 1; clustIdx <= numClusters; clustIdx++)
    {
        // find distance between each data point and nearest cluster
        value_t local_sum = 0;
        // Datapoints allocated for each process to compute
        int chunk = numData / numProcs;
        int scrap = chunk + (numData % numProcs);
        // Start index for data
        int start = chunk * rank;
        // End index of Data
        int end = start + chunk - 1;
        // Last process gets leftover datapoints
        if (rank == (numProcs - 1)) end = start + scrap - 1;
        value_t local_distances[end-start];
        dataset_t data = getDataVecFromMPIWin(start, end, numFeatures);

        for (int pointIdx = 0; pointIdx < data.size(); pointIdx++)
        {
            int dataIdx = pointIdx + start;
            local_distances[pointIdx] = nearest_MPI(data[pointIdx], dataIdx, func, numClusters);
            local_sum += local_distances[pointIdx];
        }

        // Size of each sub-array to gather
        int recLen[numProcs];
        // Index of each sub-array to gather into distances
        int disp[numProcs];
        for(int i = 0; i < numProcs; i++)
        {
            recLen[i] = chunk;
            disp[i] = i * chunk;
        }
        recLen[numProcs-1] = scrap;
        // TODO: Allgather not a great use of memory
        // Aggregate distances, distribute to all processes
        MPI_Gatherv(local_distances, (end - start + 1), MPI_FLOAT, distances, recLen, disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
        // Reduce sum, distribute to all processes
        MPI_Reduce(&local_sum, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);


        
        if(rank == 0)
        {
            dataset_t data = getDataVecFromMPIWin(0, numData - 1, numFeatures);
            // select point to be next cluster center weighted by nearest distance squared
            sum *= floatDistr();
            for (int j = 0; j < data.size(); j++)
            {
                if ((sum -= distances[j]) <= 0)
                {
                    int count = 0;
                    setClusterCoord(clusterCount, numFeatures, &data[j]);
                    setClusterCount(clusterCount, &count);
                    clusterCount++;
                    // clusters.push_back({0, datapoint_t(data[j])});
                    break;
                }
            }
        }
    }

    // Datapoints allocated for each process to compute
    int chunk = numData / numProcs;
    int scrap = chunk + (numData % numProcs);
    // Start index for data
    int start = chunk * rank;
    // End index of Data
    int end = start + chunk - 1;
    // Last process gets leftover datapoints
    if (rank == (numProcs - 1)) end = start + scrap - 1;
    value_t local_distances[end-start];
    dataset_t data = getDataVecFromMPIWin(start, end, numFeatures);

    for (int i = 0; i < data.size(); i++)
    {
        nearest_MPI(data[i], i+start, func, clusterCount);
    }

}

void Kmeans::kPlusPlus(dataset_t &data, value_t (*func)(datapoint_t &, datapoint_t &))
{
    RNGType rng(time(NULL));
    boost::uniform_int<> intRange(0, data.size());
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    value_t sum;
    std::vector<value_t> distances(data.size());

    // initialize first cluster randomly
    clusters.push_back({0, datapoint_t(data[intDistr()])});

    //initialize remaining clusters
    for (int clustIdx = 1; clustIdx < numClusters; clustIdx++)
    {
        // find distance between each data point and nearest cluster
        sum = 0;
#pragma omp parallel for shared(data, distances), schedule(static), reduction(+ \
                                                                              : sum)
        for (int pointIdx = 0; pointIdx < data.size(); pointIdx++)
        {
            distances[pointIdx] = nearest(data[pointIdx], pointIdx, func);
            sum += distances[pointIdx];
        }

        // select point to be next cluster center weighted by nearest distance squared
        sum *= floatDistr();
        for (int j = 0; j < data.size(); j++)
        {
            if ((sum -= distances[j]) <= 0)
            {
                clusters.push_back({0, datapoint_t(data[j])});
                break;
            }
        }
    }

// assign data points to nearest clusters
#pragma omp parallel for shared(data), schedule(static)
    for (int i = 0; i < data.size(); i++)
    {
        nearest(data[i], i, func);
    }
}

value_t Kmeans::nearest(datapoint_t &point, int &pointIdx, value_t (*func)(datapoint_t &, datapoint_t &))
{
    value_t tempDist, minDist = INT_MAX - 1;

    // find distance between point and all clusters
    for (int i = 0; i < clusters.size(); i++)
    {
        tempDist = std::pow(func(point, clusters[i].coords), 2);
        if (minDist > tempDist)
        {
            minDist = tempDist;
            clustering[pointIdx] = i;
        }
    }

    return minDist;
}

value_t Kmeans::nearest_MPI(datapoint_t &point, int pointIdx, value_t (*func)(datapoint_t &, datapoint_t &), int clusterCount)
{
    value_t tempDist, minDist = INT_MAX - 1;

    // find distance between point and all clusters
    for (int i = 0; i < clusterCount; i++)
    {
        datapoint_t coord = getClusterCoord(i, point.size());
        tempDist = std::pow(func(point, coord), 2);
        if (minDist > tempDist)
        {
            minDist = tempDist;
            // Put clustering number into window
            MPI_Put(&i, 1, MPI_INT, 0, pointIdx, 1, MPI_INT, clusteringWin);
        }
    }

    return minDist;
}

void Kmeans::fit(dataset_t &data, int overSampling, value_t (*func)(datapoint_t &, datapoint_t &), int initIters)
{

    int changed;
    value_t currError;
    int numFeatures = data[0].size();

    for (int run = 0; run < numRestarts; run++)
    {
        clusters = clusters_t();
        clustering = clustering_t(data.size(), -1);

        // initialize clusters with scalableKmeans algorithm
        std::vector<value_t> closestDists = scaleableKmeans(data, overSampling, func, initIters);

        do
        {
            // reinitialize clusters
            for (int i = 0; i < numClusters; i++)
            {
                clusters[i] = {0, datapoint_t(numFeatures, 0.)};
            }

            // calc sum of each feature for all points belonging to a cluster
            for (int i = 0; i < data.size(); i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    clusters[clustering[i]].coords[j] += data[i][j];
                }
                clusters[clustering[i]].count++;
            }

            // divide sum by number of points belonging to the cluster to obtain average
            for (int i = 0; i < numClusters; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    clusters[i].coords[j] /= clusters[i].count;
                }
            }

            // reassign points to cluster
            changed = 0;
#pragma omp parallel for shared(data, closestDists), schedule(static), reduction(+ \
                                                                                 : changed)
            for (int i = 0; i < data.size(); i++)
            {
                value_t dist = std::pow(func(data[i], clusters[clustering[i]].coords), 2);
                if (dist > closestDists[i] || closestDists[i] < 0)
                {

                    int before = clustering[i];
                    closestDists[i] = nearest(data[i], i, func);
                    if (before != clustering[i])
                    {
                        changed++;
                    }
                }
            }
        } while (changed > (data.size() >> 10)); // do until 99.9% of data doesnt change

        // get total sum of distances from each point to their cluster center
        currError = 0;
        for (int i = 0; i < data.size(); i++)
        {
            currError += std::pow(func(data[i], clusters[clustering[i]].coords), 2);
        }

        // if this round produced lowest error, keep clustering
        if (currError < bestError)
        {
            bestError = currError;
            bestClustering = clustering;
            bestClusters = clusters;
        }
    }
}

std::vector<value_t> Kmeans::scaleableKmeans(dataset_t &data, int &overSampling, value_t (*func)(datapoint_t &, datapoint_t &), int initIters)
{
    RNGType rng(time(NULL));
    boost::uniform_int<> intRange(0, data.size());
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    // initialize the closest distances array to large vals
    std::vector<value_t> closestDists(data.size(), INT_MAX);

    // initialize first cluster randomly
    int randIdx = intDistr();
    clusters.push_back({0, datapoint_t(std::begin(data[randIdx]), std::end(data[randIdx]))});

    // select candidate clusters
    value_t sum;
    int prevNumClusters = 0;
    for (int i = 0; i < initIters; i++)
    {
        sum = 0;
#pragma omp parallel for shared(data, closestDists), schedule(static), reduction(+ \
                                                                                 : sum)
        for (int i = 0; i < data.size(); i++)
        {
            smartClusterUpdate(data[i], i, prevNumClusters, closestDists, func);
            sum += closestDists[i];
        }
        prevNumClusters = clusters.size();

        // sample each datapoint individually to get an expectation of overSampling new clusters
        for (int j = 0; j < data.size(); j++)
        {
            if (floatDistr() < ((value_t)overSampling) * closestDists[j] / sum)
            {
                clusters.push_back({0, datapoint_t(std::begin(data[j]), std::end(data[j]))});
                clustering[j] = clusters.size() - 1;
            }
        }
    }

    // reassign points to last round of new clusters
#pragma omp parallel for shared(data), schedule(static)
    for (int i = 0; i < data.size(); i++)
    {
        smartClusterUpdate(data[i], i, prevNumClusters, closestDists, func);
    }

    // weight candidates based on how many points are in each cluster
    std::vector<int> weights(clusters.size(), 0);
    for (int i = 0; i < data.size(); i++)
    {
        weights[clustering[i]]++;
    }

    // get normalizing sum
    sum = 0;
    for (int i = 0; i < weights.size(); i++)
    {
        sum += weights[i];
    }

    // select numClusters clusters from candidates based on weights
    value_t randNum;
    clusters_t selectedClusters = clusters_t();
    clustering_t selectedClusterings = clustering_t(data.size(), -1);
    for (int i = 0; i < numClusters; i++)
    {
        randNum = floatDistr() * sum;
        for (int j = 0; j < clusters.size(); j++)
        {
            if ((randNum -= weights[j]) <= 0)
            {
                sum -= weights[j];
                selectedClusters.push_back(clusters[j]);
                clusters.erase(clusters.begin() + j);
                weights.erase(weights.begin() + j);

#pragma omp parallel for shared(data, selectedClusterings), schedule(static)
                for (int k = 0; k < data.size(); k++)
                {
                    if (clustering[k] == j)
                    {
                        selectedClusterings[k] = selectedClusters.size() - 1;
                    }
                }
                break;
            }
        }
    }

    // assign data points to nearest clusters
    clustering = selectedClusterings;
    clusters = selectedClusters;
#pragma omp parallel for shared(data, closestDists), schedule(static)
    for (int i = 0; i < data.size(); i++)
    {
        if (clustering[i] == -1)
        {
            closestDists[i] = nearest(data[i], i, func);
        }
    }

    return closestDists;
}

void Kmeans::smartClusterUpdate(datapoint_t &point, int &pointIdx, int &clusterIdx, std::vector<value_t> &distances,
                                value_t (*func)(datapoint_t &, datapoint_t &))
{
    value_t tempDist, minDist = INT_MAX - 1;
    int minDistIdx = -1;

    // find the closest new cluster to the point
    for (int i = clusterIdx; i < clusters.size(); i++)
    {
        tempDist = std::pow(func(point, clusters[i].coords), 2);
        if (tempDist < minDist)
        {
            minDist = tempDist;
            minDistIdx = i;
        }
    }

    // compare the closest new cluster to the previously closest cluster and reassign if necessary
    if (minDist < distances[pointIdx])
    {
        distances[pointIdx] = minDist;
        clustering[pointIdx] = minDistIdx;
    }
}

void Kmeans::createCoreSet(dataset_t &data, int &sampleSize, value_t (*func)(datapoint_t &, datapoint_t &))
{
    coreset.data.reserve(sampleSize);
    coreset.weights.reserve(sampleSize);

    RNGType rng(time(NULL));
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    // calculate the mean of the data
    datapoint_t mean(0, data[0].size());
    for (auto &datapoint : data)
    {
        for (int i = 0; i < datapoint.size(); i++)
        {
            mean[i] += datapoint[i];
        }
    }

    for (int i = 0; i < mean.size(); i++)
    {
        mean[i] /= data.size();
    }

    // calculate distances between the mean and all datapoints
    double distanceSum = 0;
    std::vector<value_t> distances;
    distances.reserve(data.size());
    for (int i = 0; i < data.size(); i++)
    {
        distances[i] = func(mean, data[i]);
        distanceSum += distances[i];
    }

    // calculate the distribution used to choose the datapoints to create the coreset
    value_t partOne = 0.5 * (1 / data.size()); // first portion of distribution calculation that is constant
    double sum = 0;
    std::vector<value_t> distribution;
    distribution.reserve(data.size());
    for (int i = 0; i < data.size(); i++)
    {
        distribution[i] = partOne + 0.5 * distances[i] / distanceSum;
        sum += distribution[i];
    }

    // create pointers to each datapoint in data
    std::vector<datapoint_t *> ptrData;
    ptrData.reserve(data.size());
    for (int i = 0; i < data.size(); i++)
    {
        ptrData[i] = &data[i];
    }

    // create the coreset
    double randNum;
    for (int i = 0; i < sampleSize; i++)
    {
        randNum = floatDistr() * sum;
        for (int j = 0; j < ptrData.size(); j++)
        {
            if ((randNum -= distribution[j]) <= 0)
            {
                coreset.data[i] = *ptrData[j];
                coreset.weights[i] = 1 / (sampleSize * distribution[j]);
                sum -= distribution[j];
                ptrData.erase(ptrData.begin() + j);
                distribution.erase(distribution.begin() + j);
            }
        }
    }
}

void Kmeans::setMPIWindows(MPI_Win dataWin, MPI_Win clusteringWin, MPI_Win clusterCoordWin, MPI_Win clusterCountWin)
{
    this->dataWin = dataWin;
    this->clusteringWin = clusteringWin;
    this->clusterCoordWin = clusterCoordWin;
    this->clusterCountWin = clusterCountWin;
}

dataset_t Kmeans::getDataVecFromMPIWin(int start, int end, int numFeatures)
{
    int startIdx = start * numFeatures;
    int endIdx = end * numFeatures;
    int size = endIdx - startIdx;
    value_t data[size];
    MPI_Get(data, size, MPI_FLOAT, 0, startIdx, size, MPI_FLOAT, dataWin);

    dataset_t dataVec = dataset_t((end - start + 1), datapoint_t(numFeatures));

    for(int i=0; i<dataVec.size(); i++)
    {
        for(int j=0; j<numFeatures; j++)
        {
            dataVec[i][j] = data[(i*numFeatures)+j];
        }
    }
    return dataVec;
}

datapoint_t Kmeans::getClusterCoord(int idx, int numFeatures)
{
    value_t coord[numFeatures];
    MPI_Get(coord, numFeatures, MPI_FLOAT, 0, idx*numFeatures, numFeatures, MPI_FLOAT, clusterCoordWin);

    datapoint_t coordVec(numFeatures);

    for(int i = 0; i < coordVec.size(); i++)
    {
        coordVec[i] = coord[i];
    }
    return coordVec;
}
int Kmeans::getClusterCount(int idx)
{
    int count;
    MPI_Get(&count, 1, MPI_INT, 0, idx, 1, MPI_INT, clusterCoordWin);
    return count;
}

int Kmeans::getClustering(int idx)
{
    int count;
    MPI_Get(&count, 1, MPI_INT, 0, idx, 1, MPI_INT, clusteringWin);
    return count;
}

void Kmeans::setClusterCount(int idx, int* count)
{
    MPI_Put(count, 1, MPI_INT, 0, idx, 1, MPI_INT, clusterCountWin);
}

void Kmeans::setClusterCoord(int idx, int numFeatures, datapoint_t* coord)
{
    MPI_Put(coord, coord->size(), MPI_FLOAT, 0, idx*numFeatures, coord->size(), MPI_FLOAT, clusterCoordWin);
}

bool Kmeans::setNumClusters(int numClusters)
{
    this->numClusters = numClusters;
    return true;
}

bool Kmeans::setNumRestarts(int numRestarts)
{
    this->numRestarts = numRestarts;
    return true;
}

bool Kmeans::setNumThreads(int numThreads)
{
    this->numThreads = numThreads;
    omp_set_num_threads(this->numThreads);
    return true;
}

value_t Kmeans::distanceL2(datapoint_t &p1, datapoint_t &p2)
{
    value_t sum = 0;
    for (int i = 0; i < p1.size(); i++)
    {
        sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }

    return std::sqrt(sum);
}