
#include "Lloyd.hpp"

void Lloyd::maximize(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed;
    do
    {
        updateClusters();
        changed = reassignPoints(distances, distanceFunc);

    } while (changed > (matrix->numRows * MIN_PERCENT_CHANGED)); // do until 99.9% of data doesnt change
}

void Lloyd::updateClusters()
{
    // reinitialize clusters
    std::fill(clusters->data.begin(), clusters->data.end(), 0);

    // calc the weighted sum of each feature for all points belonging to a cluster
    for (int i = 0; i < matrix->numRows; i++)
    {
        for (int j = 0; j < matrix->numCols; j++)
        {
            clusters->at(clustering->at(i), j) += weights->at(i) * matrix->at(i, j);
        }
    }

    // average out the weighted sum of each cluster based on the number of datapoints assigned to it
    for (int i = 0; i < clusters->numRows; i++)
    {
        for (int j = 0; j < clusters->numCols; j++)
        {
            clusters->data.at(i * clusters->numCols + j) /= clusterWeights->at(i);
        }
    }
}

int Lloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;
    for (int i = 0; i < matrix->numRows; i++)
    {

        // find closest cluster for each datapoint and update cluster assignment
        int before = clustering->at(i);
        auto closestCluster = findClosestCluster(i, distanceFunc);
        updateClustering(i, closestCluster.clusterIdx);
        distances->at(i) = std::pow(closestCluster.distance, 2);

        // check if cluster assignments have changed
        if (before != clustering->at(i))
        {
            changed++;
        }
    }

    return changed;
}

int OptimizedLloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;
    for (int i = 0; i < matrix->numRows; i++)
    {
        // check distance to previously closest cluster, if it increased then recalculate distances to all clusters
        value_t dist = std::pow((*distanceFunc)(&*matrix->at(i), &*clusters->at(clustering->at(i)), matrix->numCols), 2);
        if (dist > distances->at(i) || distances->at(i) < 0)
        {
            // find closest cluster for each datapoint and update cluster assignment
            int before = clustering->at(i);
            auto closestCluster = findClosestCluster(i, distanceFunc);
            updateClustering(i, closestCluster.clusterIdx);
            distances->at(i) = std::pow(closestCluster.distance, 2);

            // check if cluster assignments have changed
            if (before != clustering->at(i))
            {
                changed++;
            }
        }
        else // distance is smaller, thus update the distances vector
        {
            distances->at(i) = dist;
        }
    }

    return changed;
}

void OMPLloyd::updateClusters()
{
    // reinitialize clusters
    std::fill(clusters->data.begin(), clusters->data.end(), 0);

    // calc the weighted sum of each feature for all points belonging to a cluster
    for (int i = 0; i < matrix->numRows; i++)
    {
        for (int j = 0; j < matrix->numCols; j++)
        {
            clusters->at(clustering->at(i), j) += weights->at(i) * matrix->at(i, j);
        }
    }

    // average out the weighted sum of each cluster based on the number of datapoints assigned to it
#pragma omp parallel for schedule(static), collapse(2)
    for (int i = 0; i < clusters->numRows; i++)
    {
        for (int j = 0; j < clusters->numCols; j++)
        {
            clusters->data.at(i * clusters->numCols + j) /= clusterWeights->at(i);
        }
    }
}

int OMPLloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;

#pragma omp parallel for shared(distances, distanceFunc), schedule(static), reduction(+ \
                                                                                      : changed)
    for (int i = 0; i < matrix->numRows; i++)
    {
        // find closest cluster for each datapoint and update cluster assignment
        int before = clustering->at(i);
        auto closestCluster = findClosestCluster(i, distanceFunc);
        atomicUpdateClustering(this, i, closestCluster.clusterIdx);
        distances->at(i) = std::pow(closestCluster.distance, 2);

        // check if cluster assignments have changed
        if (before != clustering->at(i))
        {
            changed++;
        }
    }

    return changed;
}

int OMPOptimizedLloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;

#pragma omp parallel for shared(distances, distanceFunc), schedule(static), reduction(+ \
                                                                                      : changed)
    for (int i = 0; i < matrix->numRows; i++)
    {
        // check distance to previously closest cluster, if it increased then recalculate distances to all clusters
        value_t dist = std::pow((*distanceFunc)(&*matrix->at(i), &*clusters->at(clustering->at(i)), matrix->numCols), 2);
        if (dist > distances->at(i) || distances->at(i) < 0)
        {
            // find closest cluster for each datapoint and update cluster assignment
            int before = clustering->at(i);
            auto closestCluster = findClosestCluster(i, distanceFunc);
            atomicUpdateClustering(this, i, closestCluster.clusterIdx);
            distances->at(i) = std::pow(closestCluster.distance, 2);

            // check if cluster assignments have changed
            if (before != clustering->at(i))
            {
                changed++;
            }
        }
        else // distance is smaller, thus update the distances vector
        {
            distances->at(i) = dist;
        }
    }

    return changed;
}

void MPILloyd::updateClusters()
{
    std::vector<value_t> copyWeights(clusterWeights->size());
    MPI_Reduce(clusterWeights->data(), copyWeights.data(), copyWeights.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // reinitialize clusters
        std::fill(clusters->data.begin(), clusters->data.end(), 0);

        // calc the weighted sum of each feature for all points belonging to a cluster
        for (int i = 0; i < matrix->numRows; i++)
        {
            for (int j = 0; j < matrix->numCols; j++)
            {
                clusters->at(clustering->at(i), j) += weights->at(i) * matrix->at(i, j);
            }
        }

        // average out the weighted sum of each cluster based on the number of datapoints assigned to it
        for (int i = 0; i < clusters->numRows; i++)
        {
            for (int j = 0; j < clusters->numCols; j++)
            {
                clusters->data.at(i * clusters->numCols + j) /= copyWeights.at(i);
            }
        }
    }

    MPI_Bcast(clusters->data.data(), clusters->data.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

int MPILloyd::reassignPoints(std::vector<value_t> *distances, IDistanceFunctor *distanceFunc)
{
    int changed = 0;

    for (int i = 0; i < matrixChunk->numRows; i++)
    {
        // find closest cluster for each datapoint and update cluster assignment
        int before = clustering->at(i);
        auto closestCluster = findClosestCluster(&*matrixChunk->at(i), distanceFunc);
        AbstractMPIKmeansAlgorithm::updateClustering(this, displacements->at(rank) + i, closestCluster.clusterIdx);
        distances->at(displacements->at(rank) + i) = std::pow(closestCluster.distance, 2);

        // check if cluster assignments have changed
        if (before != clustering->at(i))
        {
            changed++;
        }
    }

    MPI_Allgatherv(MPI_IN_PLACE, lengths->at(rank), MPI_INT, clustering->data(),
                   lengths->data(), displacements->data(), MPI_INT, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, lengths->at(rank), MPI_INT, distances->data(),
                   lengths->data(), displacements->data(), MPI_INT, MPI_COMM_WORLD);

    return changed;
}

void MPILloyd::setUp(BundledAlgorithmData *bundledData)
{
    AbstractMPIKmeansAlgorithm::setUp(this, dynamic_cast<BundledMPIAlgorithmData *>(bundledData));
}