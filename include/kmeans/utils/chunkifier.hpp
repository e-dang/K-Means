#pragma once

#include <mpi.h>

#include <cstdint>
#include <kmeans/types/parallelism.hpp>
#include <kmeans/utils/utils.hpp>

namespace hpkmeans
{
template <Parallelism Level>
class Chunkifier
{
public:
    Chunkifier() : m_rank(0), m_numProcs(0), m_totalNumData(0), m_lengths(), m_displacements() {}

    Chunkifier(const int32_t totalNumData) :
        m_rank(0), m_numProcs(1), m_totalNumData(totalNumData), m_lengths(), m_displacements()
    {
        initialize();
    }

    inline const int rank() const noexcept { return m_rank; }

    inline const int numProcs() const noexcept { return m_numProcs; }

    inline const int32_t totalNumData() const noexcept { return m_totalNumData; }

    inline const std::vector<int32_t>& lengths() const noexcept { return m_lengths; }

    inline const int32_t& lengthsAt(const int rank) const noexcept { return m_lengths[rank]; }

    inline const int32_t myLength() const noexcept { return m_lengths[m_rank]; }

    inline const std::vector<int32_t>& displacements() const noexcept { return m_displacements; }

    inline const int32_t myDisplacement() const noexcept { return m_displacements[m_rank]; }

private:
    template <Parallelism _Level = Level>
    inline std::enable_if_t<isSharedMemory(_Level)> initialize()
    {
        m_lengths       = std::vector<int32_t>(1, m_totalNumData);
        m_displacements = std::vector<int32_t>(1, 0);
    }

    template <Parallelism _Level = Level>
    inline std::enable_if_t<isDistributed(_Level)> initialize()
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &m_numProcs);
        MPI_Allreduce(MPI_IN_PLACE, &m_totalNumData, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        int32_t chunk = m_totalNumData / m_numProcs;
        int32_t scrap = chunk + (m_totalNumData % m_numProcs);

        m_lengths       = std::vector<int32_t>(m_numProcs);  // size of each sub-array to gather
        m_displacements = std::vector<int32_t>(m_numProcs);  // index of each sub-array to gather
        for (int i = 0; i < m_numProcs; ++i)
        {
            m_lengths[i]       = chunk;
            m_displacements[i] = i * chunk;
        }
        m_lengths[m_numProcs - 1] = scrap;
    }

private:
    int m_rank;
    int m_numProcs;
    int32_t m_totalNumData;
    std::vector<int32_t> m_lengths;
    std::vector<int32_t> m_displacements;
};
}  // namespace hpkmeans