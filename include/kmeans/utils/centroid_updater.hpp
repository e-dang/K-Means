#pragma once

#include <mpi.h>

#include <kmeans/types/parallelism.hpp>
#include <kmeans/utils/utils.hpp>

namespace hpkmeans
{
template <typename T, Parallelism Level>
class CentroidUpdater
{
public:
    template <Parallelism _Level = Level>
    std::enable_if_t<isSharedMemory(_Level)> updateCentroids(const Matrix<T>* const data, Matrix<T>* const centroids,
                                                             const VectorView<int32_t>* const assignments,
                                                             const std::vector<T>* const weights,
                                                             std::vector<T>* const clusterWeights) const
    {
        calcClusterWeights(centroids, assignments, weights, clusterWeights);
        calcNewCentroids(data, centroids, assignments, weights, clusterWeights);
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level)> updateCentroids(const Matrix<T>* const data, Matrix<T>* const centroids,
                                                            const VectorView<int32_t>* const assignments,
                                                            const std::vector<T>* const weights,
                                                            std::vector<T>* const clusterWeights) const
    {
        calcClusterWeights(centroids, assignments, weights, clusterWeights);
        MPI_Allreduce(MPI_IN_PLACE, clusterWeights->data(), clusterWeights->size(), matchMPIType<T>(), MPI_SUM,
                      MPI_COMM_WORLD);

        calcNewCentroids(data, centroids, assignments, weights, clusterWeights);
    }

private:
    void calcClusterWeights(Matrix<T>* const centroids, const VectorView<int32_t>* const assignments,
                            const std::vector<T>* const weights, std::vector<T>* const clusterWeights) const
    {
        if (weights == nullptr)
            calcClusterCounts(centroids, assignments, clusterWeights);
        else
            calcClusterWeightsImpl(centroids, assignments, weights, clusterWeights);
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isSingleThreaded(_Level)> calcClusterCounts(Matrix<T>* const,
                                                                 const VectorView<int32_t>* const assignments,
                                                                 std::vector<T>* const clusterWeights) const
    {
        for (int32_t i = 0; i < assignments->viewSize(); ++i)
        {
            ++clusterWeights->at(assignments->at(i));
        }
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isMultiThreaded(_Level)> calcClusterCounts(Matrix<T>* const centroids,
                                                                const VectorView<int32_t>* const assignments,
                                                                std::vector<T>* const clusterWeights) const
    {
#pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < centroids->numRows(); ++i)
        {
            for (int32_t j = 0; j < assignments->viewSize(); ++j)
            {
                if (i == assignments->at(j))
                    ++clusterWeights->at(i);
            }
        }
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isSingleThreaded(_Level)> calcClusterWeightsImpl(Matrix<T>* const,
                                                                      const VectorView<int32_t>* const assignments,
                                                                      const std::vector<T>* const weights,
                                                                      std::vector<T>* const clusterWeights) const
    {
        for (int32_t i = 0; i < assignments->viewSize(); ++i)
        {
            clusterWeights->at(assignments->at(i)) += weights->at(i);
        }
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isMultiThreaded(_Level)> calcClusterWeightsImpl(Matrix<T>* const centroids,
                                                                     const VectorView<int32_t>* const assignments,
                                                                     const std::vector<T>* const weights,
                                                                     std::vector<T>* const clusterWeights) const
    {
#pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < centroids->numRows(); ++i)
        {
            for (int32_t j = 0; j < assignments->viewSize(); ++j)
            {
                if (i == assignments->at(j))
                    clusterWeights->at(i) += weights->at(j);
            }
        }
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isSharedMemory(_Level)> calcNewCentroids(const Matrix<T>* const data, Matrix<T>* const centroids,
                                                              const VectorView<int32_t>* const assignments,
                                                              const std::vector<T>* const weights,
                                                              std::vector<T>* const clusterWeights) const
    {
        calcClusterSums(data, centroids, assignments, weights);
        normalizeCentroidsSums(centroids, clusterWeights);
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level)> calcNewCentroids(const Matrix<T>* const data, Matrix<T>* const centroids,
                                                             const VectorView<int32_t>* const assignments,
                                                             const std::vector<T>* const weights,
                                                             std::vector<T>* const clusterWeights) const
    {
        calcClusterSums(data, centroids, assignments, weights);

        MPI_Allreduce(MPI_IN_PLACE, centroids->data(), centroids->size(), matchMPIType<T>(), MPI_SUM, MPI_COMM_WORLD);

        normalizeCentroidsSums(centroids, clusterWeights);
    }

    inline void calcClusterSums(const Matrix<T>* const data, Matrix<T>* const centroids,
                                const VectorView<int32_t>* const assignments, const std::vector<T>* const weights) const
    {
        if (weights == nullptr)
            calcUnWeightedClusterSums(data, centroids, assignments);
        else
            calcWeightedClusterSums(data, centroids, assignments, weights);
    }

    inline void calcUnWeightedClusterSums(const Matrix<T>* const data, Matrix<T>* const centroids,
                                          const VectorView<int32_t>* const assignments) const
    {
        for (int32_t i = 0; i < data->numRows(); ++i)
        {
            for (int32_t j = 0; j < data->cols(); ++j)
            {
                centroids->at(assignments->at(i), j) += data->at(i, j);
            }
        }
    }

    inline void calcWeightedClusterSums(const Matrix<T>* const data, Matrix<T>* const centroids,
                                        const VectorView<int32_t>* const assignments,
                                        const std::vector<T>* const weights) const
    {
        for (int32_t i = 0; i < data->numRows(); ++i)
        {
            for (int32_t j = 0; j < data->cols(); ++j)
            {
                centroids->at(assignments->at(i), j) += weights->at(i) * data->at(i, j);
            }
        }
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isSingleThreaded(_Level)> normalizeCentroidsSums(Matrix<T>* const centroids,
                                                                      std::vector<T>* const clusterWeights) const
    {
        for (int32_t i = 0; i < centroids->numRows(); ++i)
        {
            for (int32_t j = 0; j < centroids->cols(); ++j)
            {
                centroids->at(i, j) /= clusterWeights->at(i);
            }
        }
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isMultiThreaded(_Level)> normalizeCentroidsSums(Matrix<T>* const centroids,
                                                                     std::vector<T>* const clusterWeights) const
    {
#pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < centroids->numRows(); ++i)
        {
            for (int32_t j = 0; j < centroids->cols(); ++j)
            {
                centroids->at(i, j) /= clusterWeights->at(i);
            }
        }
    }
};
}  // namespace hpkmeans