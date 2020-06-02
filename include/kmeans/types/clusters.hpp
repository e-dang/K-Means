#pragma once

#include <kmeans/types/parallelism.hpp>
#include <limits>
#include <matrix/matrix.hpp>
#include <numeric>

namespace hpkmeans
{
template <typename T>
class Clusters
{
public:
    Clusters() :
        p_data(nullptr),
        p_weights(nullptr),
        m_assignments(),
        m_sqDistances(),
        m_centroids(),
        m_clusterCounts(),
        m_error(std::numeric_limits<T>::max())
    {
    }

    Clusters(const int32_t numClusters, const Matrix<T>* data, const std::vector<T>* weights = nullptr) :
        p_data(data),
        p_weights(weights),
        m_assignments(data->numRows(), -1),
        m_sqDistances(data->numRows(), std::numeric_limits<T>::max()),
        m_centroids(numClusters, data->cols()),
        m_clusterCounts(numClusters, 0),
        m_error(std::numeric_limits<T>::max())
    {
        validateWeights();
    }

    bool operator<(const Clusters& lhs) const { return m_error < lhs.m_error; }

    bool operator>(const Clusters& lhs) const { return m_error > lhs.m_error; }

    void addCentroid(const int32_t dataIdx)
    {
        m_centroids.append(p_data->crowBegin(dataIdx), p_data->crowEnd(dataIdx));
    }

    template <Parallelism Level>
    void updateCentroids()
    {
        m_centroids.fill(0.0);
        std::fill(m_clusterCounts.begin(), m_clusterCounts.end(), 0);
        calcClusterCounts<Level>();
        updateCentroidsImpl<Level>();
    }

    template <class AssignmentUpdater>
    void updateAssignments(const AssignmentUpdater& updater)
    {
        updater.update(p_data, &m_centroids, &m_assignments, &m_sqDistances);
    }

    template <Parallelism Level>
    std::enable_if_t<Level == Parallelism::Serial> calcError()
    {
        m_error = std::accumulate(m_sqDistances.cbegin(), m_sqDistances.cend(), 0.0);
    }

    template <Parallelism Level>
    std::enable_if_t<Level == Parallelism::OMP> calcError()
    {
        T cost = 0.0;

#pragma omp parallel for schedule(static), reduction(+ : cost)
        for (int32_t i = 0; i < static_cast<int32_t>(m_sqDistances.size()); ++i)
        {
            cost += m_sqDistances[i];
        }

        m_error = cost;
    }

    const std::vector<int32_t>* const assignments() const { return &m_assignments; }

    const std::vector<T>* const sqDistances() const { return &m_sqDistances; }

    int32_t size() const { return m_centroids.numRows(); }

    int32_t maxSize() const { return m_centroids.rows(); }

    const Matrix<T>* const getCentroids() const { return &m_centroids; }

    const std::vector<int32_t>* const getClustering() const { return &m_assignments; }

    const T getError() const { return m_error; }

private:
    void validateWeights()
    {
        if (p_weights != nullptr && p_data->numRows() != static_cast<int32_t>(p_weights->size()))
            throw std::length_error("The data and corresponding weights must have the same number of entries!");
    }

    template <Parallelism Level>
    std::enable_if_t<Level == Parallelism::Serial> updateCentroidsImpl()
    {
        for (int32_t i = 0; i < p_data->numRows(); ++i)
        {
            for (int32_t j = 0; j < p_data->cols(); ++j)
            {
                // m_centroids.at(m_assignments[i], j) += p_weights->at(i) * p_data->at(i, j);
                m_centroids.at(m_assignments[i], j) += p_data->at(i, j);
            }
        }

        for (int32_t i = 0; i < m_centroids.numRows(); ++i)
        {
            for (int32_t j = 0; j < m_centroids.cols(); ++j)
            {
                m_centroids.at(i, j) /= m_clusterCounts[i];
            }
        }
    }

    template <Parallelism Level>
    std::enable_if_t<Level == Parallelism::OMP> updateCentroidsImpl()
    {
        // #pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < p_data->numRows(); ++i)
        {
            for (int32_t j = 0; j < p_data->cols(); ++j)
            {
                m_centroids.at(m_assignments[i], j) += p_weights->at(i) * p_data->at(i, j);
            }
        }

#pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < m_centroids.numRows(); ++i)
        {
            for (int32_t j = 0; j < m_centroids.cols(); ++j)
            {
                m_centroids.at(i, j) /= m_clusterCounts[i];
            }
        }
    }

    template <Parallelism Level>
    std::enable_if_t<Level == Parallelism::Serial> calcClusterCounts()
    {
        for (int i = 0; i < m_centroids.numRows(); ++i)
        {
            for (int j = 0; j < m_assignments.size(); ++j)
            {
                if (i == m_assignments[j])
                    ++m_clusterCounts[i];
            }
        }
    }

    template <Parallelism Level>
    std::enable_if_t<Level == Parallelism::OMP> calcClusterCounts()
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < m_centroids.numRows(); ++i)
        {
            for (int j = 0; j < m_assignments.size(); ++j)
            {
                if (i == m_assignments[j])
                    ++m_clusterCounts[i];
            }
        }
    }

private:
    const Matrix<T>* p_data;
    const std::vector<T>* p_weights;
    std::vector<int32_t> m_assignments;
    std::vector<T> m_sqDistances;
    Matrix<T> m_centroids;
    std::vector<int32_t> m_clusterCounts;
    T m_error;
};
}  // namespace hpkmeans