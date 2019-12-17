#include "MPIKmeans.hpp"
#include "mpi.h"
#include "math.h"
#include <assert.h>

MPIKmeans::MPIKmeans(int numClusters, int numRestarts) : numClusters(numClusters), numRestarts(numRestarts)
{
    bestError = -1;
}

MPIKmeans::~MPIKmeans()
{
}

void MPIKmeans::fit(int numData, int numFeatures, value_t *data, value_t (*func)(datapoint_t &, datapoint_t &))
{
    RNGType rng(time(NULL));
    boost::uniform_int<> intRange(0, numData);
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    initMPIMembers(numData, numFeatures, data);

    int changed;
    value_t currError;

    for (int run = 0; run < numRestarts; run++)
    {
        // initialize clusters with k++ algorithm
        kPlusPlus(numData, numFeatures, data, func, intDistr, floatDistr);

        do
        {
            if (rank == 0)
            {
                // reinitialize clusters
                for (int i = 0; i < numClusters; i++)
                {
                    for (auto &coord : clusterCoord_MPI)
                    {
                        coord.assign(coord.size(), 0.);
                    }
                    clusterCount_MPI.assign(clusterCount_MPI.size(), 0);
                    // clusters[i] = {0, datapoint_t(numFeatures, 0.)};
                }

                // calc sum of each feature for all points belonging to a cluster
                for (int i = 0; i < numData; i++)
                {
                    for (int j = 0; j < numFeatures; j++)
                    {
                        clusterCoord_MPI[clustering_MPI[i]][j] += data[(i * numFeatures) + j];
                        // clusters[clustering[i]].coords[j] +=  data[i][j];
                    }
                    clusterCount_MPI[clustering_MPI[i]]++;
                    // clusters[clustering[i]].count++;
                }

                // divide sum by number of points belonging to the cluster to obtain average
                for (int i = 0; i < numClusters; i++)
                {
                    for (int j = 0; j < numFeatures; j++)
                    {
                        clusterCoord_MPI[i][j] /= clusterCount_MPI[i];
                        // clusters[i].coords[j] /= clusters[i].count;
                    }
                }
            }
            // Sync clusters between processes
            MPI_Bcast(clusterCount_MPI.data(), clusterCount_MPI.size(), MPI_INT, 0, MPI_COMM_WORLD);
            // TODO: could do this in one call but requries flattening into c-array
            for (int i = 0; i < clusterCoord_MPI.size(); i++)
            {
                MPI_Bcast(clusterCoord_MPI[i].data(), clusterCoord_MPI[i].size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
            }

            // reassign points to cluster
            changed = 0;
            int local_changed = 0;
            for (int i = 0; i < data_MPI.size(); i++)
            {
                int before = clusteringChunk_MPI[i];
                nearest(data_MPI[i], i, func, numClusters);
                if (before != clusteringChunk_MPI[i])
                {
                    local_changed++;
                }
            }

            // Aggregate changed
            MPI_Allreduce(&local_changed, &changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        } while (changed > (numData >> 10)); // do until 99.9% of data doesnt change

        // Gather clustering
        MPI_Allgatherv(clusteringChunk_MPI.data(), vLens_MPI[rank], MPI_INT, clustering_MPI.data(),
                       vLens_MPI.data(), vDisps_MPI.data(), MPI_INT, MPI_COMM_WORLD);

        // get total sum of distances from each point to their cluster center
        currError = 0;
        value_t localError = 0;

        for (int i = 0; i < data_MPI.size(); i++)
        {
            int idx = i + startIdx_MPI;
            localError += std::pow(func(data_MPI[i], clusterCoord_MPI[clustering_MPI[idx]]), 2);
            // localError += std::pow(func(data[i], clusters[clustering[i]].coords), 2);
        }

        MPI_Reduce(&localError, &currError, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            // if this round produced lowest error, keep clustering
            if (currError < bestError || bestError < 0)
            {
                bestError = currError;
                bestClustering = clustering_MPI;
                bestClusterCount = clusterCount_MPI;
                bestClusterCoord = clusterCoord_MPI;
            }
        }
    }
}

void MPIKmeans::fit(int numData, int numFeatures, value_t *data, int overSampling, value_t (*func)(datapoint_t &, datapoint_t &), int initIters)
{

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    initMPIMembers(numData, numFeatures, data);

    RNGType rng(time(NULL));
    boost::uniform_int<> intRange(0, data_MPI.size());
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    int changed;
    value_t currError;

    for (int run = 0; run < numRestarts; run++)
    {
        // initialize clusters with k++ algorithm
        std::vector<value_t> closestDists = scaleableKmeans(numData, numFeatures, data, overSampling, func, intDistr, floatDistr, initIters);

        do
        {
            if (rank == 0)
            {
                // reinitialize clusters
                for (int i = 0; i < numClusters; i++)
                {
                    for (auto &coord : clusterCoord_MPI)
                    {
                        coord.assign(coord.size(), 0.);
                    }
                    clusterCount_MPI.assign(clusterCount_MPI.size(), 0);
                }

                // calc sum of each feature for all points belonging to a cluster
                for (int i = 0; i < numData; i++)
                {
                    for (int j = 0; j < numFeatures; j++)
                    {
                        clusterCoord_MPI[clustering_MPI[i]][j] += data[(i * numFeatures) + j];
                    }
                    clusterCount_MPI[clustering_MPI[i]]++;
                }

                // divide sum by number of points belonging to the cluster to obtain average
                for (int i = 0; i < numClusters; i++)
                {
                    for (int j = 0; j < numFeatures; j++)
                    {
                        clusterCoord_MPI[i][j] /= clusterCount_MPI[i];
                    }
                }
            }
            // Sync clusters between processes
            MPI_Bcast(clusterCount_MPI.data(), clusterCount_MPI.size(), MPI_INT, 0, MPI_COMM_WORLD);
            // TODO: could do this in one call but requries flattening into c-array
            for (int i = 0; i < clusterCoord_MPI.size(); i++)
            {
                MPI_Bcast(clusterCoord_MPI[i].data(), clusterCoord_MPI[i].size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
            }

            // reassign points to cluster
            changed = 0;
            int local_changed = 0;
            for (int i = 0; i < data_MPI.size(); i++)
            {
                value_t dist = std::pow(func(data_MPI[i], clusterCoord_MPI[clusteringChunk_MPI[i]]), 2);
                if (dist > closestDists[i] || closestDists[i] < 0)
                {
                    int before = clusteringChunk_MPI[i];
                    closestDists[i] = nearest(data_MPI[i], i, func, numClusters);
                    if (before != clusteringChunk_MPI[i])
                    {
                        local_changed++;
                    }
                }
            }

            // Aggregate changed
            MPI_Allreduce(&local_changed, &changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        } while (changed > (numData >> 10)); // do until 99.9% of data doesnt change

        // Gather clustering
        MPI_Allgatherv(clusteringChunk_MPI.data(), vLens_MPI[rank], MPI_INT, clustering_MPI.data(),
                       vLens_MPI.data(), vDisps_MPI.data(), MPI_INT, MPI_COMM_WORLD);

        // get total sum of distances from each point to their cluster center
        currError = 0;
        value_t localError = 0;

        for (int i = 0; i < data_MPI.size(); i++)
        {
            int idx = i + startIdx_MPI;
            localError += std::pow(func(data_MPI[i], clusterCoord_MPI[clustering_MPI[idx]]), 2);
        }

        MPI_Reduce(&localError, &currError, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            // if this round produced lowest error, keep clustering
            if (currError < bestError || bestError < 0)
            {
                bestError = currError;
                bestClustering = clustering_MPI;
                bestClusterCount = clusterCount_MPI;
                bestClusterCoord = clusterCoord_MPI;
            }
        }
    }
}

void MPIKmeans::kPlusPlus(int numData, int numFeatures, value_t *data, value_t (*func)(datapoint_t &, datapoint_t &), boost::variate_generator<RNGType, boost::uniform_int<>> intDistr, boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr)
{
    int rank, numProcs, clusterCount = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    value_t sum = 0;
    std::vector<value_t> distances(numData);

    if (rank == 0)
    {
        // initialize first cluster randomly
        int randIdx = intDistr();
        clusterCount_MPI[0] = 0;
        clusterCoord_MPI[0] = datapoint_t(data + (randIdx * numFeatures), data + ((randIdx * numFeatures) + numFeatures));
        clusterCount++;
    }

    std::vector<value_t> local_distances(data_MPI.size());
    bcastClusters(clusterCount);

    for (int clustIdx = 1; clustIdx < numClusters; clustIdx++)
    {
        // find distance between each data point and nearest cluster
        value_t local_sum = 0;

        for (int pointIdx = 0; pointIdx < data_MPI.size(); pointIdx++)
        {
            local_distances[pointIdx] = nearest(data_MPI[pointIdx], pointIdx, func, clusterCount);
            local_sum += local_distances[pointIdx];
        }
        // Aggregate distances, distribute to all processes
        MPI_Gatherv(local_distances.data(), vLens_MPI[rank], MPI_FLOAT, distances.data(),
                    vLens_MPI.data(), vDisps_MPI.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        // Reduce sum, distribute to all processes
        MPI_Reduce(&local_sum, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            // select point to be next cluster center weighted by nearest distance squared
            sum *= floatDistr();
            for (int j = 0; j < numData; j++)
            {
                if ((sum -= distances[j]) <= 0)
                {
                    clusterCount_MPI[clusterCount] = 0;

                    for (int i = 0; i < numFeatures; i++)
                    {
                        clusterCoord_MPI[clusterCount][i] = data[(j * numFeatures) + i];
                    }
                    clusterCount++;
                    break;
                }
            }
        }

        bcastClusters(clusterCount);
    }

    for (int i = 0; i < data_MPI.size(); i++)
    {
        nearest(data_MPI[i], i, func, clusterCount);
    }

    MPI_Allgatherv(clusteringChunk_MPI.data(), vLens_MPI[rank], MPI_INT, clustering_MPI.data(),
                   vLens_MPI.data(), vDisps_MPI.data(), MPI_INT, MPI_COMM_WORLD);
}

std::vector<value_t> MPIKmeans::scaleableKmeans(int &numData, int &numFeatures, value_t *data, int &overSampling,
                                                value_t (*func)(datapoint_t &, datapoint_t &),
                                                boost::variate_generator<RNGType, boost::uniform_int<>> intDistr,
                                                boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr,
                                                int initIters)
{
    int rank, numProcs, prevNumClusters = 0, clusterCount = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // initialize the closest distances array to large vals
    std::vector<value_t> closestDists(numData, INT_MAX);
    std::vector<value_t> local_distances(data_MPI.size(), INT_MAX);
    std::vector<int> lengths(numProcs, 0);
    std::vector<int> localCounts;
    dataset_t localCoords;

    int randIdx = intDistr();
    localCounts.push_back(0);
    localCoords.push_back(data_MPI[randIdx]);
    clusterCount++;

    // select candidate clusters
    value_t local_sum; //, sum = 0;
    for (int i = 0; i < initIters; i++)
    {
        local_sum = 0;
        for (int i = 0; i < data_MPI.size(); i++)
        {
            smartClusterUpdate(data_MPI[i], i, prevNumClusters, clusterCount, local_distances.data(), localCoords, func);
            local_sum += local_distances[i];
        }
        prevNumClusters = clusterCount;

        // sample each datapoint individually to get an expectation of overSampling new clusters
        for (int j = 0; j < data_MPI.size(); j++)
        {
            if (floatDistr() < ((value_t)overSampling) * local_distances[j] / local_sum)
            {
                localCounts.push_back(0);
                localCoords.push_back(data_MPI[j]);
                clusterCount++;
            }
        }
    }

    // reassign points to last round of new clusters
    for (int i = 0; i < data_MPI.size(); i++)
    {
        smartClusterUpdate(data_MPI[i], i, prevNumClusters, clusterCount, local_distances.data(), localCoords, func);
    }

    MPI_Allgather(&clusterCount, 1, MPI_INT, lengths.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int sum = 0;
    for (auto &val : lengths)
    {
        sum += val;
    }

    if (rank == 0)
    {
        clusterCoord_MPI.resize(sum);
        for (int i = 0; i < sum; i++)
        {
            clusterCoord_MPI[i].resize(numFeatures);
        }
        int start = lengths[0];

        for (int i = 0; i < start; i++)
        {
            clusterCoord_MPI[i] = localCoords[i];
        }

        for (int i = 1; i < numProcs; i++)
        {
            for (int j = 0; j < lengths[i]; j++, start++)
            {
                assert(clusterCoord_MPI[start].size() == numFeatures);
                MPI_Recv(clusterCoord_MPI[start].data(), numFeatures, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    else
    {
        for (int i = 0; i < lengths[rank]; i++)
        {

            MPI_Send(localCoords[i].data(), numFeatures, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
        clusterCoord_MPI.resize(numClusters);
        for (int i = 0; i < numClusters; i++)
        {
            clusterCoord_MPI[i].resize(numFeatures);
        }
    }

    MPI_Gatherv(clusteringChunk_MPI.data(), vLens_MPI[rank], MPI_INT, clustering_MPI.data(),
                vLens_MPI.data(), vDisps_MPI.data(), MPI_INT, 0, MPI_COMM_WORLD);

    // weight candidates based on how many points are in each cluster
    if (rank == 0)
    {
        std::vector<int> weights(clusterCoord_MPI.size(), 0);
        for (int i = 0; i < numData; i++)
        {
            weights[clustering_MPI[i]]++;
        }

        // get normalizing sum
        sum = 0;
        for (int i = 0; i < weights.size(); i++)
        {
            sum += weights[i];
        }

        // select numClusters clusters from candidates based on weights
        value_t randNum;
        dataset_t selectedClusterCoords = dataset_t();
        clustering_t selectedClusterings = clustering_t(numData, -1);
        for (int i = 0; i < numClusters; i++)
        {
            randNum = floatDistr() * sum;
            for (int j = 0; j < clusterCoord_MPI.size(); j++)
            {
                if ((randNum -= weights[j]) <= 0)
                {
                    sum -= weights[j];
                    selectedClusterCoords.push_back(clusterCoord_MPI[j]);
                    clusterCoord_MPI.erase(clusterCoord_MPI.begin() + j);
                    weights.erase(weights.begin() + j);

                    for (int k = 0; k < numData; k++)
                    {
                        if (clustering_MPI[k] == j)
                        {
                            selectedClusterings[k] = selectedClusterCoords.size() - 1;
                        }
                    }
                    break;
                }
            }
        }
        clusterCoord_MPI = selectedClusterCoords;
        clustering_MPI = selectedClusterings;
        clusterCount = clusterCoord_MPI.size();
    }

    for (int i = 0; i < clusterCoord_MPI.size(); i++)
    {
        MPI_Bcast(clusterCoord_MPI[i].data(), clusterCoord_MPI[i].size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    MPI_Scatterv(clustering_MPI.data(), vLens_MPI.data(), vDisps_MPI.data(), MPI_INT, clusteringChunk_MPI.data(),
                 vLens_MPI[rank], MPI_INT, 0, MPI_COMM_WORLD);

    // assign data points to nearest clusters
    for (int i = 0; i < data_MPI.size(); i++)
    {
        local_distances[i] = nearest(data_MPI[i], i, func, numClusters);
    }

    MPI_Allgatherv(local_distances.data(), vLens_MPI[rank], MPI_FLOAT, closestDists.data(),
                   vLens_MPI.data(), vDisps_MPI.data(), MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(clusteringChunk_MPI.data(), vLens_MPI[rank], MPI_INT, clustering_MPI.data(),
                   vLens_MPI.data(), vDisps_MPI.data(), MPI_INT, MPI_COMM_WORLD);

    return closestDists;
}

value_t MPIKmeans::nearest(datapoint_t &point, int pointIdx, value_t (*func)(datapoint_t &, datapoint_t &), int clusterCount)
{
    value_t tempDist, minDist = INT_MAX - 1;

    // find distance between point and all clusters
    for (int i = 0; i < clusterCount; i++)
    {
        tempDist = std::pow(func(point, clusterCoord_MPI[i]), 2);
        if (minDist > tempDist)
        {
            minDist = tempDist;
            clusteringChunk_MPI[pointIdx] = i;
        }
    }
    return minDist;
}

void MPIKmeans::smartClusterUpdate(datapoint_t &point, int &pointIdx, int &prevNumClusters, int &clusterCount,
                                   value_t *distances, dataset_t &localCoords, value_t (*func)(datapoint_t &, datapoint_t &))
{
    value_t tempDist, minDist = INT_MAX - 1;
    int minDistIdx = -1;

    // find the closest new cluster to the point
    for (int i = prevNumClusters; i < clusterCount; i++)
    {
        tempDist = std::pow(func(point, localCoords[i]), 2);
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
        clusteringChunk_MPI[pointIdx] = minDistIdx;
    }
}

void MPIKmeans::initMPIMembers(int numData, int numFeatures, value_t *data)
{
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // Set data start/end indices

    // Datapoints allocated for each process to compute
    int chunk = numData / numProcs;
    int scrap = chunk + (numData % numProcs);
    // Start index for data
    startIdx_MPI = chunk * rank;
    // End index of Data
    endIdx_MPI = startIdx_MPI + chunk - 1;
    // Last process gets leftover datapoints
    if (rank == (numProcs - 1))
        endIdx_MPI = startIdx_MPI + scrap - 1;

    int size = endIdx_MPI - startIdx_MPI + 1;

    // Resize member vectors
    clusterCount_MPI.resize(numClusters);
    clusterCoord_MPI.resize(numClusters);
    for (int i = 0; i < clusterCoord_MPI.size(); i++)
    {
        clusterCoord_MPI[i].resize(numFeatures);
    }
    clustering_MPI.resize(numData);
    clusteringChunk_MPI.resize(size);

    // Size of each sub-array to gather
    vLens_MPI.resize(numProcs);
    // Index of each sub-array to gather
    vDisps_MPI.resize(numProcs);
    for (int i = 0; i < numProcs; i++)
    {
        vLens_MPI[i] = chunk;
        vDisps_MPI[i] = i * chunk;
    }
    vLens_MPI[numProcs - 1] = scrap;

    // Create disp/len arrays for data scatter
    int dataLens[numProcs];
    int dataDisps[numProcs];
    for (int i = 0; i < numProcs; i++)
    {
        dataLens[i] = vLens_MPI[i] * numFeatures;
        dataDisps[i] = vDisps_MPI[i] * numFeatures;
    }

    value_t tempData[size * numFeatures];

    // scatter data
    if (rank == 0)
    {
        assert(data != NULL);
        MPI_Scatterv(data, dataLens, dataDisps, MPI_FLOAT, tempData, dataLens[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatterv(NULL, dataLens, dataDisps, MPI_FLOAT, tempData, dataLens[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // convert to dataset_t
    data_MPI = arrayToDataset(tempData, size, numFeatures);
}

dataset_t MPIKmeans::arrayToDataset(value_t *data, int size, int numFeatures)
{
    dataset_t dataVec = dataset_t(size, datapoint_t(numFeatures));

    for (int i = 0; i < dataVec.size(); i++)
    {
        for (int j = 0; j < numFeatures; j++)
        {
            dataVec[i][j] = data[(i * numFeatures) + j];
        }
    }
    return dataVec;
}

void MPIKmeans::bcastClusters(int &clusterCount)
{
    MPI_Bcast(clusterCount_MPI.data(), clusterCount_MPI.size(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&clusterCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < clusterCoord_MPI.size(); i++)
    {
        MPI_Bcast(clusterCoord_MPI[i].data(), clusterCoord_MPI[i].size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
}