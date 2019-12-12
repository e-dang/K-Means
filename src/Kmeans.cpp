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

void Kmeans::initMPIMembers(int numData, int numFeatures, value_t *data)
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

void Kmeans::fit(dataset_t &data, value_t (*func)(datapoint_t &, datapoint_t &))
{

    int changed;
    int numData = data.size();
    int numFeatures = data[0].size();
    value_t currError;

    for (int run = 0; run < numRestarts; run++)
    {
        clusters = clusters_t();
        clustering = clustering_t(numData, -1);

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

void Kmeans::fit_coreset(value_t (*func)(datapoint_t &, datapoint_t &))
{
    int changed;
    int numData = coreset.data.size();
    int numFeatures = coreset.data[0].size();
    value_t currError;

    for (int run = 0; run < numRestarts; run++)
    {
        bestClusters = clusters_t();
        bestClustering = clustering_t(numData, -1);
        clusters = clusters_t();
        clustering = clustering_t(numData, -1);
        // initialize clusters with k++ algorithm
        kPlusPlus(coreset.data, func); 

        do
        {
            // reinitialize clusters
            for (int i = 0; i < numClusters; i++)
            {
                clusters[i] = {0, datapoint_t(numFeatures, 0.)};
            }

            // calc the weighted sum of each feature for all points belonging to a cluster
            std::vector<value_t> cluster_weights(clusters.size(), 0.0);
            for (int i = 0; i < numData; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    clusters[clustering[i]].coords[j] += coreset.weights[i] * coreset.data[i][j];
                }
                clusters[clustering[i]].count++;
                cluster_weights.at(clustering[i]) += coreset.weights[i];
            }

            // divide the sum of the points by the total cluster weight to obtain average
            for (int i = 0; i < numClusters; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    clusters[i].coords[j] /= cluster_weights[i];
                }
            }

            // reassign points to cluster
            changed = 0;
#pragma omp parallel for schedule(static), reduction(+ \
                                                                   : changed)
            for (int i = 0; i < numData; i++)
            {
                int before = clustering[i];
                nearest(coreset.data[i], i, func);
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
            // std::cout << currError << std::endl;
            currError += std::pow(func(coreset.data[i], clusters[clustering[i]].coords), 2);
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

void Kmeans::fit_MPI_win(int numData, int numFeatures, value_t (*func)(datapoint_t &, datapoint_t &))
{
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int changed;
    value_t currError;

    if (rank == 0)
    {
        clusters = clusters_t(numClusters);
        clustering = clustering_t(numData);
        // clustering = clustering_t(numData, -1);
    }

    for (int run = 0; run < numRestarts; run++)
    {
        // initialize clusters with k++ algorithm
        kPlusPlus_MPI_win(numData, numFeatures, func);

        do
        {
            if (rank == 0)
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
                    datapoint_t coord = getClusterCoord(clst, numFeatures);
                    datapoint_t data = getDataVecFromMPIWin(i, i + 1, numFeatures)[0];
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
            if (rank == (numProcs - 1))
                end = start + scrap - 1;
            dataset_t data = getDataVecFromMPIWin(start, end, numFeatures);

            for (int i = 0; i < data.size(); i++)
            {
                int before, current;
                MPI_Get(&before, 1, MPI_INT, 0, i + start, 1, MPI_INT, clusteringWin);
                nearest_MPI_win(data[i], i + start, func, numClusters);
                MPI_Get(&current, 1, MPI_INT, 0, i + start, 1, MPI_INT, clusteringWin);
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
        if (rank == (numProcs - 1))
            end = start + scrap - 1;
        dataset_t data = getDataVecFromMPIWin(start, end, numFeatures);

        for (int i = 0; i < data.size(); i++)
        {
            int idx = i + start;
            int clst = getClustering(idx);
            datapoint_t coord = getClusterCoord(clst, numFeatures);
            localError += std::pow(func(data[i], coord), 2);
            // localError += std::pow(func(data[i], clusters[clustering[i]].coords), 2);
        }

        MPI_Reduce(&localError, &currError, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            // if this round produced lowest error, keep clustering
            if (currError < bestError)
            {
                bestError = currError;

                // Fill best clustering from shared window
                MPI_Get(&bestClustering[0], numData, MPI_INT, 0, numData, 1, MPI_INT, clusteringWin);

                // Fill best clusters from shared window
                for (int i = 0; i < numClusters; i++)
                {
                    bestClusters[i].count = getClusterCount(i);
                    bestClusters[i].coords = getClusterCoord(i, numFeatures);
                }
            }
        }
    }
}

void Kmeans::fit_MPI(int numData, int numFeatures, value_t *data, value_t (*func)(datapoint_t &, datapoint_t &))
{
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    initMPIMembers(numData, numFeatures, data);

    int changed;
    value_t currError;

    for (int run = 0; run < numRestarts; run++)
    {
        // initialize clusters with k++ algorithm
        kPlusPlus_MPI(numData, numFeatures, data, func);

        do
        {
            if (rank == 0)
            {
                // reinitialize clusters
                for (int i = 0; i < numClusters; i++)
                {
                    for (auto coord : clusterCoord_MPI)
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
                nearest_MPI(data_MPI[i], i, func, numClusters);
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
            if (currError < bestError)
            {
                bestError = currError;
                bestClustering = clustering_MPI;
                bestClusterCount = clusterCount_MPI;
                bestClusterCoord = clusterCoord_MPI;
            }
        }
    }
}

void Kmeans::kPlusPlus_MPI_win(int numData, int numFeatures, value_t (*func)(datapoint_t &, datapoint_t &))
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

    if (rank == 0)
    {
        int ranNum = intDistr();
        dataset_t randomDataPoint = getDataVecFromMPIWin(ranNum, ranNum + 1, numFeatures);
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
        if (rank == (numProcs - 1))
            end = start + scrap - 1;
        value_t local_distances[end - start];
        dataset_t data = getDataVecFromMPIWin(start, end, numFeatures);

        for (int pointIdx = 0; pointIdx < data.size(); pointIdx++)
        {
            int dataIdx = pointIdx + start;
            local_distances[pointIdx] = nearest_MPI_win(data[pointIdx], dataIdx, func, numClusters);
            local_sum += local_distances[pointIdx];
        }

        // Size of each sub-array to gather
        int recLen[numProcs];
        // Index of each sub-array to gather into distances
        int disp[numProcs];
        for (int i = 0; i < numProcs; i++)
        {
            recLen[i] = chunk;
            disp[i] = i * chunk;
        }
        recLen[numProcs - 1] = scrap;
        // TODO: Allgather not a great use of memory
        // Aggregate distances, distribute to all processes
        MPI_Gatherv(local_distances, (end - start + 1), MPI_FLOAT, distances, recLen, disp, MPI_FLOAT, 0, MPI_COMM_WORLD);
        // Reduce sum, distribute to all processes
        MPI_Reduce(&local_sum, &sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
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
    if (rank == (numProcs - 1))
        end = start + scrap - 1;
    value_t local_distances[end - start];
    dataset_t data = getDataVecFromMPIWin(start, end, numFeatures);

    for (int i = 0; i < data.size(); i++)
    {
        nearest_MPI_win(data[i], i + start, func, clusterCount);
    }
}

void Kmeans::kPlusPlus_MPI(int numData, int numFeatures, value_t *data, value_t (*func)(datapoint_t &, datapoint_t &))
{
    int rank, numProcs, clusterCount = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    RNGType rng(time(NULL));
    boost::uniform_int<> intRange(0, data_MPI.size());
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    value_t sum = 0;
    std::vector<value_t> distances(numData);

    if (rank == 0)
    {
        // initialize first cluster randomly
        int ranNum = intDistr();
        clusterCount_MPI[0] = 0;
        clusterCoord_MPI[0] = data_MPI[intDistr()];
        clusterCount++;
        // clusters.push_back({0, datapoint_t(randomDataPoint[0])});
    }
    std::vector<value_t> local_distances(data_MPI.size());

    for (int clustIdx = 1; clustIdx < numClusters; clustIdx++)
    {
        // find distance between each data point and nearest cluster
        value_t local_sum = 0;

        for (int pointIdx = 0; pointIdx < data_MPI.size(); pointIdx++)
        {
            local_distances[pointIdx] = nearest_MPI(data_MPI[pointIdx], pointIdx, func, clusterCount);
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
    }
    MPI_Bcast(clusterCount_MPI.data(), clusterCount_MPI.size(), MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < clusterCoord_MPI.size(); i++)
    {
        MPI_Bcast(clusterCoord_MPI[i].data(), clusterCoord_MPI[i].size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    for (int i = 0; i < data_MPI.size(); i++)
    {
        nearest_MPI(data_MPI[i], i, func, clusterCount);
    }

    MPI_Allgatherv(clusteringChunk_MPI.data(), vLens_MPI[rank], MPI_INT, clustering_MPI.data(),
                   vLens_MPI.data(), vDisps_MPI.data(), MPI_INT, MPI_COMM_WORLD);
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

value_t Kmeans::nearest_MPI_win(datapoint_t &point, int pointIdx, value_t (*func)(datapoint_t &, datapoint_t &), int clusterCount)
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

value_t Kmeans::nearest_MPI(datapoint_t &point, int pointIdx, value_t (*func)(datapoint_t &, datapoint_t &), int clusterCount)
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

    RNGType rng(time(NULL));
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    // calculate the mean of the data
    datapoint_t mean(data[0].size(), 0);
    auto mean_data = mean.data();

#pragma omp parallel for shared(data), schedule(static), reduction(+ : mean_data[: data[0].size()])
    for (int nth_datapoint = 0; nth_datapoint < data.size(); nth_datapoint++)
    {
        for (int i = 0; i < data[0].size(); i++)
        {
            mean_data[i] += data[nth_datapoint][i];
        }
    }

    for (int i = 0; i < mean.size(); i++)
    {
        mean[i] /= data.size();
    }

    // calculate distances between the mean and all datapoints
    double distanceSum = 0;
    std::vector<value_t> distances(data.size(), 0);
#pragma omp parallel for shared(data, distances), schedule(static), reduction(+ : distanceSum) 
    for (int i = 0; i < data.size(); i++)
    {
        distances[i] = func(mean, data[i]);
        distanceSum += distances[i];
    }

    // calculate the distribution used to choose the datapoints to create the coreset
    value_t partOne = 0.5 * (1.0 / (float)data.size()); // first portion of distribution calculation that is constant
    double sum = 0.0;
    std::vector<value_t> distribution(data.size(),0);
#pragma omp parallel for shared(distribution, distances), schedule(static), reduction(+ : sum) 
    for (int i = 0; i < data.size(); i++)
    {
        distribution[i] = partOne + 0.5 * distances[i] / distanceSum;
        sum += distribution[i];
    }

    // create pointers to each datapoint in data
    std::vector<datapoint_t *> ptrData(data.size());
#pragma omp parallel for shared(data, ptrData), schedule(static) // this section might have false sharing, which will degrade performance
    for (int i = 0; i < data.size(); i++)
    {
        ptrData[i] = &data[i];
    }

    // create the coreset
    double randNum;
    std::vector<datapoint_t> c(sampleSize);
    std::vector<value_t> w(sampleSize, 0);
    coreset.data = c;
    coreset.weights = w;
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
                break;
            }
        }
    }
}

void Kmeans::createCoreSet_MPI(int numData, int numFeatures, value_t *data, int sampleSize, value_t (*func)(datapoint_t &, datapoint_t &)){
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // scatter the data across all processes
    initMPIMembers(numData, numFeatures, data);

    RNGType rng(time(NULL));
    boost::uniform_int<> intRange(0, data_MPI.size());
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);
    
    // compute the mean, sum, and sqd sum of the data local to each machine
    datapoint_t local_mean(data_MPI[0].size(), 0);
    datapoint_t local_sum(data_MPI[0].size(), 0);
    datapoint_t local_sqd_sum(data_MPI[0].size(), 0);

    auto local_mean_data = local_mean.data();
    auto local_sum_data = local_sum.data();
    auto local_sqd_sum_data = local_sqd_sum.data();

    for (int nth_datapoint = 0; nth_datapoint < data_MPI.size(); nth_datapoint++)
    {
        // std::cout << rank << " rank's datapoint: "; 
        for (int mth_feature = 0; mth_feature < data_MPI[0].size(); mth_feature++)
        {
            local_mean_data[mth_feature] += data_MPI[nth_datapoint][mth_feature];
            local_sum_data[mth_feature] +=  data_MPI[nth_datapoint][mth_feature];
            local_sqd_sum_data[mth_feature] += std::pow(data_MPI[nth_datapoint][mth_feature], 2);
            // std::cout<< data_MPI[nth_datapoint][mth_feature] << "\t";
        }
    }
    // std::cout << local_sum[0] << "\t" << local_sqd_sum[0] <<std::endl;

    for (int i = 0; i < local_mean.size(); i++)
    {
        local_mean[i] /= data_MPI.size();
        // std::cout << local_mean[i] << std::endl;
    }
    int local_cardinality = data_MPI.size();

    // gather the local sums, squared sums, and means onto central machine
    float *mean = NULL;
    float *sum = NULL;
    float *sqd_sum = NULL;
    int *local_cardinalities = NULL;
    if (rank == 0){
        mean = (float*)malloc(numProcs*data_MPI[0].size()*sizeof(float));
        sum = (float*)malloc(numProcs*data_MPI[0].size()*sizeof(float));
        sqd_sum =(float*) malloc(numProcs*data_MPI[0].size()*sizeof(float));
        local_cardinalities = (int*)malloc(numProcs*sizeof(int));
    }

    MPI_Gather(local_mean.data(), data_MPI[0].size(), MPI_FLOAT, mean, data_MPI[0].size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_sum.data(), data_MPI[0].size(), MPI_FLOAT, sum, data_MPI[0].size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_sqd_sum.data(), data_MPI[0].size(), MPI_FLOAT, sqd_sum, data_MPI[0].size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_cardinality, 1, MPI_INT, local_cardinalities, 1, MPI_INT, 0, MPI_COMM_WORLD);


    // root machine will compute the following:
    std::vector<value_t> global_mean(data_MPI[0].size(),0);
    std::vector<value_t> quant_errs(numProcs, 0);
    std::vector<int> uniform_sample_counts(numProcs,0); // number of points to be sampled uniformly on the ith machine
    std::vector<int> phi_sample_counts(numProcs,0); // number of points to be sampled based on quant error on the ith machine
    std::vector<int> samples_per_proc(numProcs, 0); // Elements represent number of data points to be sampled per machine. Index corresponds to rank of the proccess. 
    std::vector<int> samples_per_proc_disp(numProcs, 0);
    std::vector<int> data_per_proc(numProcs, 0); // Elements represent the amount of data (number of floating point numbers) to sample on each machine (element-wise sum of the above two vectors)*dataset dimensionality
    std::vector<int> data_per_proc_disp(numProcs, 0);
    int dataset_cardinality = 0;
    float total_quant_err = 0.0;

    if (rank == 0){
        // compute the global mean from the gathered mean data
        std::vector<datapoint_t> local_means;
        for (int nth_proc = 0; nth_proc <numProcs; nth_proc++){
            for (int mth_feature = 0; mth_feature < data_MPI[0].size(); mth_feature++){
                global_mean[mth_feature] += mean[(nth_proc*data_MPI[0].size()) + mth_feature];
                // std::cout << mean[(nth_proc*data_MPI[0].size()) + mth_feature] << std::endl;
            }
            std::vector<value_t> loc_mean(mean[nth_proc*data_MPI[0].size()], mean[nth_proc*data_MPI[0].size() + data_MPI[0].size()]);
            local_means.push_back(loc_mean);
        }
        for (int mth_feature = 0; mth_feature < data_MPI[0].size(); mth_feature++){
            global_mean[mth_feature] /= (float)numProcs;
            // std::cout << global_mean[mth_feature] << std::endl;
        }

        // compute the total cardinality of the dataset -- might be unecessary but doing this to make this section of code robust to chagne in the way we get a dataset
        for (int nth_proc = 0; nth_proc < numProcs; nth_proc++){
            dataset_cardinality += local_cardinalities[nth_proc];
            // std::cout << "local_cardinalities: " << local_cardinalities[nth_proc] << std::endl;
        }
        // std::cout << "dataset cardinality: "<< dataset_cardinality << std::endl;

        // compute the local quantization error for each machine and the total quantization error
        for (int nth_proc = 0; nth_proc < numProcs; nth_proc++){
            value_t ith_quant_err;
            // for (int mth_feature = 0; mth_feature < data_MPI[0].size(); mth_feature++){
            //     ith_quant_err += sqd_sum[(nth_proc*data_MPI[0].size())+mth_feature] - 2*global_mean[mth_feature]*sum[(nth_proc*data_MPI[0].size())+mth_feature] + local_cardinalities[nth_proc]*std::pow(global_mean[mth_feature],2);
            // }
            ith_quant_err = std::pow(func(global_mean, local_means[nth_proc]),2);
            quant_errs[nth_proc] = ith_quant_err;
            // std::cout<< "ith_quant_err: " << ith_quant_err << std::endl;
            total_quant_err += ith_quant_err;
        }
        // std::cout << "total_quant_err: " << total_quant_err << std::endl;

        // compute the number of points to sample from each machine
        double randNum;
        for (int nth_sample = 0; nth_sample < sampleSize; nth_sample ++){
            randNum = floatDistr();
            if (randNum > .5){
                randNum = floatDistr()*dataset_cardinality;
                int cumsum = 0;
                for (int nth_proc = 0; nth_proc < numProcs; nth_proc++){
                    cumsum += local_cardinalities[nth_proc];
                    if (cumsum >= randNum) {
                        uniform_sample_counts[nth_proc] += 1;
                        samples_per_proc[nth_proc] += 1;
                        data_per_proc[nth_proc] += data_MPI[0].size();
                        break;
                    }
                }
            }
            else {
                randNum = floatDistr()*total_quant_err;
                float cumsum = 0;
                for (int nth_proc = 0; nth_proc < numProcs; nth_proc++){
                    cumsum += quant_errs[nth_proc];
                    if (cumsum >= randNum) {
                        phi_sample_counts[nth_proc] += 1;
                        samples_per_proc[nth_proc] += 1;
                        data_per_proc[nth_proc] += data_MPI[0].size();
                        break;
                    }
                }
            }
        }

        // calculate the displacement of the coreset data indices in the soon-to-be aggregated coreset array
        // std::cout << "SAMPLES PER PROC PHI: " << phi_sample_counts[0] << " "<< phi_sample_counts[1] << std::endl;
        // std::cout << "SAMPLES PER PROC UNIFORM: " << uniform_sample_counts[0] << " "<< uniform_sample_counts[1] << std::endl;
        // std::cout << "SAMPLES PER PROC: " << samples_per_proc[0] << std::endl;
        // std::cout << "SAMPLES PER PROC: " << samples_per_proc[1] << std::endl;
        for (int nth_proc = 1; nth_proc < numProcs; nth_proc++){
            data_per_proc_disp[nth_proc] = data_per_proc[nth_proc-1] + data_per_proc_disp[nth_proc];
            samples_per_proc_disp[nth_proc] = samples_per_proc[nth_proc-1] + samples_per_proc_disp[nth_proc];
        }
    }   
    // need barrier here?
    // MPI_Barrier(MPI_COMM_WORLD);
    // broadcast the global mean, total quantization error, sampling counts and local quantization errors
    MPI_Bcast(global_mean.data(), data_MPI[0].size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_quant_err, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dataset_cardinality, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(quant_errs.data(), numProcs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(phi_sample_counts.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(uniform_sample_counts.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);

    // calculate the distribution used to choose the datapoints to create the coreset
    // std::cout << "process number: " << rank <<" has this size "<< data_MPI.size() << std::endl;
    value_t partOne = 0.5 * (1.0 / dataset_cardinality); // first portion of distribution calculation that is constant
    std::vector<value_t> distribution(data_MPI.size(),0);
    std::vector<value_t> sqd_distances(data_MPI.size(),0);
    float s = 0;
    for (int i = 0; i < data_MPI.size(); i++)
    {
        sqd_distances[i] = std::pow(func(data_MPI[i], global_mean), 2);
        distribution[i] = partOne + .5*sqd_distances[i]/(dataset_cardinality*total_quant_err);
        // std::cout << "asdf" << func(data_MPI[i], global_mean) << std::endl;
        // std::cout << distribution[i] << std::endl;
        s += distribution[i];
    }
    // std::cout << s << std::endl;

    // create pointers to each datapoint in data
    std::vector<datapoint_t *> ptrData(data_MPI.size());
    for (int i = 0; i < data_MPI.size(); i++)
    {
        ptrData[i] = &data_MPI[i];
    }
    // std::cout << "helloooooo" << std::endl;
    // generate the coreset by first sampling the appropriate number of points for a given machine from the uniform distribution
    int randNum;
    double phiDist = total_quant_err; 
    double uniformDist = data_MPI.size() -1; 
    int nth_uniform_sample;
    std::vector<datapoint_t> c(uniform_sample_counts[rank] + phi_sample_counts[rank]);
    std::vector<value_t> w(uniform_sample_counts[rank] + phi_sample_counts[rank], 0);
    for (nth_uniform_sample = 0; nth_uniform_sample < uniform_sample_counts[rank]; nth_uniform_sample++){
        randNum = (int) std::round(floatDistr()*(uniformDist - nth_uniform_sample)); // subtract the contribution of the previously sampled point from the uniform dist
        // if (randNum > distribution.size()){
            // std::cout << "RandNum: " << randNum << " Size: " << distribution.size() << std::endl;
        // }
        c[nth_uniform_sample] = *ptrData[randNum];
        w[nth_uniform_sample] = (float)1.0 / (sampleSize * distribution[randNum]);
        // std::cout << sampleSize * distribution[randNum] << std::endl;
        // std::cout <<nth_uniform_sample<< " "<< w[nth_uniform_sample] << std::endl;
        phiDist -= sqd_distances[nth_uniform_sample]; // subtract contribution of the sampled point from the phi distrubition
        ptrData.erase(ptrData.begin() + randNum);
        distribution.erase(distribution.begin() + randNum);
    }
    // std::cout <<rank << "hello8  " << std::endl;
    double randPhi;
    for (int i = uniform_sample_counts[rank]; i < uniform_sample_counts[rank] + phi_sample_counts[rank]; i++)
    {
        randPhi = floatDistr() * phiDist;
        for (int j = 0; j < ptrData.size(); j++)
        {
            if ((randPhi -= sqd_distances[i]) <= 0)
            {
                c[i] = *ptrData[j];
                w[i] = (float)1.0 / (sampleSize * distribution[j]);
                // std::cout << i << " "<< w[i] << std::endl;
                // for (int j = 0; j < data_MPI[0].size(); j++){
                //     std::cout<< c[i][j] << "\t";
                // }
                // std::cout << std::endl;
                phiDist -= sqd_distances[i];
                ptrData.erase(ptrData.begin() + j);
                distribution.erase(distribution.begin() + j);
                break;
            }
        }
    }
    // std::cout<< rank<< "coreset cardinality: " << c.size() << std::endl;
    // for (int i = 0; i < c.size(); i++){
    //     for (int j = 0; j < data_MPI[0].size(); j++){
    //         std::cout << c[i][j] << "\t";
    //     }
    //     std::cout << std::endl;
    // }
    // for (int i = 0; i < w.size(); i++){
    //     std::cout << rank << " " << w[i] << std::endl;
    // }
    // flatten the 2d vector of coreset data
    // float* flattenedCoresetData = (float*) malloc(sampleSize*data_MPI[0].size()*sizeof(float));
    int local_coreset_cardinality = c.size();
    std::vector<value_t> flattenedCoresetData(local_coreset_cardinality*data_MPI[0].size());
    // std::cout <<rank<<"flattened array size: "<< local_coreset_cardinality*data_MPI[0].size() << std::endl;

    for (int i = 0; i < local_coreset_cardinality; i++){
            // std::cout <<"i: "<< i << " dim: " << dim << std::endl;
        for (int j = 0; j < data_MPI[0].size(); j++){
            // std::cout << i*data_MPI.size() + j << std::endl;
            flattenedCoresetData[i*data_MPI[0].size() + j] = c[i][j];

        // for (int j = 0; j < data_MPI[0].size(); j++){
        //         std::cout<< flattenedCoresetData[i*data_MPI[0].size() + j] << "\t";
        // }
        // std::cout << std::endl;
        }
    }
    // std::cout <<rank << "hello10" << std::endl;

    //gather the coresets back onto the root machine
    value_t* coreset_temp = NULL;
    value_t* weights_temp = NULL;

    if (rank == 0) {
        coreset_temp = (value_t*)malloc(sampleSize*data_MPI[0].size()*sizeof(float));
        weights_temp = (value_t*)malloc(sampleSize*sizeof(float));
    }
    // std::cout <<rank << "hello10" << std::endl;

    int data_send_count = data_MPI[0].size()*local_coreset_cardinality;
    // std::cout <<rank <<"data send count: "<< data_send_count << std::endl;
    // std::cout <<rank <<"recv buffer size: "<< sampleSize*data_MPI[0].size() << std::endl;
    // std::cout <<rank <<"elements reserved for data from rank 0: "<< data_per_proc[0] << std::endl;
    // std::cout <<rank <<"elements reserved for data from rank 1: "<< data_per_proc[1] << std::endl;
    // std::cout <<rank <<"displacement for data from rank 0: "<< data_per_proc_disp[0] << std::endl;
    // std::cout <<rank <<"displacement for data from rank 1: "<< data_per_proc_disp[1] << std::endl;


    MPI_Gatherv(flattenedCoresetData.data(), data_send_count, MPI_FLOAT, coreset_temp, data_per_proc.data(), data_per_proc_disp.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    // std::cout <<rank << "hello11" << std::endl;

    MPI_Gatherv(w.data(), local_coreset_cardinality, MPI_FLOAT, weights_temp, samples_per_proc.data(), samples_per_proc_disp.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    // std::cout <<rank << "hello12" << std::endl;

    if (rank == 0){
        // organize the flattened coresets_temp and weights_temp arrays into a coresets_t struct
        std::vector<datapoint_t> coreset_data;
        for (int i = 0; i < sampleSize; i ++){
            datapoint_t datapoint_temp(coreset_temp + i*data_MPI[0].size(), coreset_temp+ (i+1)*data_MPI[0].size());
            coreset_data.push_back(datapoint_temp);
            // for (int j = 0; j < data_MPI[0].size(); j++){
            //     std::cout<< coreset_temp[i*data_MPI[0].size() + j] << "\t";
            // }
            // std::cout << std::endl;
        }
        coreset.data = coreset_data;

        std::vector<float> w(weights_temp, weights_temp+sampleSize);
        coreset.weights = w;
        // for (int i = 0; i < sampleSize; i++){
        //     std::cout << weights_temp[i] << std::endl;
        // }
    }
}

void Kmeans::setMPIWindows(MPI_Win dataWin, MPI_Win clusteringWin, MPI_Win clusterCoordWin, MPI_Win clusterCountWin)
{
    this->dataWin = dataWin;
    this->clusteringWin = clusteringWin;
    this->clusterCoordWin = clusterCoordWin;
    this->clusterCountWin = clusterCountWin;
}

dataset_t Kmeans::arrayToDataset(value_t *data, int size, int numFeatures)
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

dataset_t Kmeans::getDataVecFromMPIWin(int start, int end, int numFeatures)
{
    int startIdx = start * numFeatures;
    int endIdx = end * numFeatures;
    int size = endIdx - startIdx;
    value_t data[size];
    MPI_Get(data, size, MPI_FLOAT, 0, startIdx, size, MPI_FLOAT, dataWin);

    dataset_t dataVec = dataset_t((end - start + 1), datapoint_t(numFeatures));

    for (int i = 0; i < dataVec.size(); i++)
    {
        for (int j = 0; j < numFeatures; j++)
        {
            dataVec[i][j] = data[(i * numFeatures) + j];
        }
    }
    return dataVec;
}

datapoint_t Kmeans::getClusterCoord(int idx, int numFeatures)
{
    value_t coord[numFeatures];
    MPI_Get(coord, numFeatures, MPI_FLOAT, 0, idx * numFeatures, numFeatures, MPI_FLOAT, clusterCoordWin);

    datapoint_t coordVec(numFeatures);

    for (int i = 0; i < coordVec.size(); i++)
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

void Kmeans::setClusterCount(int idx, int *count)
{
    MPI_Put(count, 1, MPI_INT, 0, idx, 1, MPI_INT, clusterCountWin);
}

void Kmeans::setClusterCoord(int idx, int numFeatures, datapoint_t *coord)
{
    MPI_Put(coord, coord->size(), MPI_FLOAT, 0, idx * numFeatures, coord->size(), MPI_FLOAT, clusterCoordWin);
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