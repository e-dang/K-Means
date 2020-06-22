#pragma once

#include <hpkmeans/types/parallelism.hpp>
#include <hpkmeans/types/vector_view.hpp>
#include <hpkmeans/utils/chunkifier.hpp>
#include <hpkmeans/utils/utils.hpp>
#include <limits>
#include <matrix/matrix.hpp>
#include <numeric>

namespace hpkmeans
{
template <typename T, Parallelism Level>
class Clusters
{
public:
    Clusters() :
        m_chunkifier(0),
        p_data(nullptr),
        p_weights(nullptr),
        m_assignments(),
        m_sqDistances(),
        m_centroids(),
        m_clusterWeights(),
        m_error(std::numeric_limits<T>::max())
    {
    }

    Clusters(const int32_t numClusters, const Matrix<T>* data, const std::vector<T>* weights = nullptr) :
        m_chunkifier(data->numRows()),
        p_data(data),
        p_weights(weights),
        m_assignments(m_chunkifier.totalNumData(), m_chunkifier.myLength(), m_chunkifier.myDisplacement(), -1),
        m_sqDistances(m_chunkifier.totalNumData(), m_chunkifier.myLength(), m_chunkifier.myDisplacement(),
                      std::numeric_limits<T>::max()),
        m_centroids(numClusters, data->cols()),
        m_clusterWeights(numClusters, 0),
        m_error(std::numeric_limits<T>::max())
    {
        validateWeights();
    }

    Clusters(const Clusters& other) :
        m_chunkifier(),
        p_data(),
        p_weights(),
        m_assignments(),
        m_sqDistances(),
        m_centroids(),
        m_clusterWeights(),
        m_error(std::numeric_limits<T>::max())
    {
        *this = other;
    }

    Clusters(Clusters&& other) = default;

    Clusters<T, Level>& operator=(const Clusters<T, Level>& rhs)
    {
        if (this != &rhs)
        {
            if (p_data != rhs.p_data)
            {
                p_data           = rhs.p_data;
                p_weights        = rhs.p_weights;
                m_assignments    = rhs.m_assignments;
                m_sqDistances    = rhs.m_sqDistances;
                m_centroids      = rhs.m_centroids;
                m_clusterWeights = rhs.m_clusterWeights;
            }
            else
            {
                std::copy(rhs.m_assignments.cbegin(), rhs.m_assignments.cend(), m_assignments.begin());
                std::copy(rhs.m_sqDistances.cbegin(), rhs.m_sqDistances.cend(), m_sqDistances.begin());
                std::copy(rhs.m_centroids.cbegin(), rhs.m_centroids.cend(), m_centroids.begin());
                std::copy(rhs.m_clusterWeights.cbegin(), rhs.m_clusterWeights.cend(), m_clusterWeights.begin());
            }

            m_chunkifier = rhs.m_chunkifier;
            m_error      = rhs.m_error;
        }

        return *this;
    }

    Clusters<T, Level>& operator=(Clusters<T, Level>&& rhs) = default;

    bool operator<(const Clusters& lhs) const { return m_error < lhs.m_error; }

    bool operator>(const Clusters& lhs) const { return m_error > lhs.m_error; }

    void clear()
    {
        std::fill(m_sqDistances.begin(), m_sqDistances.end(), std::numeric_limits<T>::max());
        m_centroids.resize(0);
    }

    void addCentroid(const int32_t dataIdx)
    {
        auto displacedDataIdx = dataIdx - m_chunkifier.myDisplacement();
        m_centroids.append(p_data->crowBegin(displacedDataIdx), p_data->crowEnd(displacedDataIdx));
    }

    void reserveCentroidSpace() { m_centroids.resize(m_centroids.numRows() + 1); }

    template <class CentroidUpdater>
    void updateCentroids(const CentroidUpdater updater)
    {
        m_centroids.fill(0.0);
        std::fill(m_clusterWeights.begin(), m_clusterWeights.end(), 0.0);
        updater.updateCentroids(p_data, &m_centroids, &m_assignments, p_weights, &m_clusterWeights);
    }

    template <class AssignmentUpdater>
    void updateAssignments(const AssignmentUpdater updater)
    {
        updater->update(p_data, &m_centroids, &m_assignments, &m_sqDistances);
    }

    void calcError()
    {
        if constexpr (isDistributed(Level))
            gatherSqDistances();

        m_error = accumulate<Level>(&m_sqDistances);

        if constexpr (isDistributed(Level))
            MPI_Bcast(&m_error, 1, matchMPIType<T>(), 0, MPI_COMM_WORLD);
    }

    void copyCentroids(const Clusters<T, Level>& other)
    {
        if (maxSize() != other.maxSize() || m_centroids.cols() != other.m_centroids.cols())
            throw std::length_error(
              "Cannot copy centroids from one clusters object to another when the dimensions aren't the same!");

        m_centroids = other.m_centroids;
    }

    int32_t size() const { return m_centroids.numRows(); }

    int32_t maxSize() const { return m_centroids.rows(); }

    const VectorView<int32_t>* const assignments() const { return &m_assignments; }

    const VectorView<T>* const sqDistances() const { return &m_sqDistances; }

    const Matrix<T>* const getCentroids() const { return &m_centroids; }

    const T getError() const { return m_error; }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level), const std::vector<int32_t>&> lengths() const
    {
        return m_chunkifier.lengths();
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level), const std::vector<int32_t>&> displacements() const
    {
        return m_chunkifier.displacements();
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level), const int32_t> totalNumData() const
    {
        return m_chunkifier.totalNumData();
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level)> bcastCentroids(const int rank)
    {
        MPI_Bcast(&*m_centroids.rowBegin(size() - 1), m_centroids.cols(), matchMPIType<T>(), rank, MPI_COMM_WORLD);
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level)> allGatherAssignments()
    {
        MPI_Allgatherv(MPI_IN_PLACE, m_chunkifier.myLength(), MPI_INT, m_assignments.data(), lengths().data(),
                       displacements().data(), MPI_INT, MPI_COMM_WORLD);
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level)> gatherAssignments(const int rank = 0)
    {
        auto displacement = m_chunkifier.myDisplacement();
        MPI_Gatherv(m_assignments.data() + displacement, m_chunkifier.myLength(), MPI_INT, m_assignments.data(),
                    lengths().data(), displacements().data(), MPI_INT, rank, MPI_COMM_WORLD);
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level)> gatherSqDistances(const int rank = 0)
    {
        auto displacement = m_chunkifier.myDisplacement();
        MPI_Gatherv(m_sqDistances.data() + displacement, m_chunkifier.myLength(), matchMPIType<T>(),
                    m_sqDistances.data(), lengths().data(), displacements().data(), matchMPIType<T>(), rank,
                    MPI_COMM_WORLD);
    }

private:
    void validateWeights()
    {
        if (p_weights != nullptr && p_data->numRows() != static_cast<int32_t>(p_weights->size()))
            throw std::length_error("The data and corresponding weights must have the same number of entries!");
    }

private:
    Chunkifier<Level> m_chunkifier;
    const Matrix<T>* p_data;
    const std::vector<T>* p_weights;
    VectorView<int32_t> m_assignments;
    VectorView<T> m_sqDistances;
    Matrix<T> m_centroids;
    std::vector<T> m_clusterWeights;
    T m_error;
};
}  // namespace hpkmeans