#pragma once

#include <hpkmeans/utils/mpi_class.hpp>
#include <vector>

namespace HPKmeans
{
template <typename int_size>
class AbstractDataChunks
{
protected:
    int m_Rank;
    int m_NumProcs;
    int_size m_TotalNumData;
    std::vector<int_size> m_Lengths;
    std::vector<int_size> m_Displacements;

public:
    AbstractDataChunks() = default;

    virtual ~AbstractDataChunks() = default;

    AbstractDataChunks(AbstractDataChunks&& other) :
        m_Rank(0), m_NumProcs(0), m_TotalNumData(0), m_Lengths(), m_Displacements()
    {
        *this = std::move(other);
    }

    AbstractDataChunks& operator=(AbstractDataChunks&& rhs)
    {
        if (this != &rhs)
        {
            rank          = rhs.rank;
            numProcs      = rhs.numProcs;
            lengths       = std::move(rhs.lengths);
            displacements = std::move(rhs.displacements);
        }

        return *this;
    }

    inline const int rank() noexcept { return m_Rank; }

    inline const int numProcs() noexcept { return m_NumProcs; }

    inline const int totalNumData() noexcept { return m_TotalNumData; }

    inline const std::vector<int_size>& lengths() const noexcept { return m_Lengths; }

    inline const int_size* lengthsData() const { return m_Lengths.data(); }

    inline const int_size& lengthsAt(const int rank) const { return m_Lengths[rank]; }

    inline const int_size& myLength() const { return m_Lengths[m_Rank]; }

    inline const std::vector<int_size>& displacements() const noexcept { return m_Displacements; }

    inline const int_size* displacementsData() const noexcept { return m_Displacements.data(); }

    inline const int_size& displacementsAt(const int rank) const { return m_Displacements[rank]; }

    inline const int_size& myDisplacement() const { return m_Displacements[m_Rank]; }

protected:
    inline virtual void initialize(const int_size numData) = 0;
};

template <typename int_size>
class SharedMemoryDataChunks : public AbstractDataChunks<int_size>
{
public:
    SharedMemoryDataChunks(const int_size numData) : AbstractDataChunks<int_size>() { initialize(numData); }

    ~SharedMemoryDataChunks() = default;

protected:
    inline void initialize(const int_size numData) override
    {
        this->m_Rank          = 0;
        this->m_NumProcs      = 1;
        this->m_TotalNumData  = numData;
        this->m_Lengths       = std::vector<int_size>(1, numData);
        this->m_Displacements = std::vector<int_size>(1, 0);
    }
};

template <typename int_size>
class MPIDataChunks : public AbstractDataChunks<int_size>, public MPIClass<float, int_size>
{
private:
    using MPIClass<float, int_size>::mpi_int_size;

public:
    MPIDataChunks(const int_size numData) : AbstractDataChunks<int_size>() { initialize(numData); }

    ~MPIDataChunks() = default;

    inline void calcChunks()
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &this->m_Rank);
        MPI_Comm_size(MPI_COMM_WORLD, &this->m_NumProcs);

        // number of datapoints allocated for each process to compute
        int_size chunk = this->m_TotalNumData / this->m_NumProcs;
        int_size scrap = chunk + (this->m_TotalNumData % this->m_NumProcs);

        this->m_Lengths       = std::vector<int_size>(this->m_NumProcs);  // size of each sub-array to gather
        this->m_Displacements = std::vector<int_size>(this->m_NumProcs);  // index of each sub-array to gather
        for (int i = 0; i < this->m_NumProcs; ++i)
        {
            this->m_Lengths.at(i)       = chunk;
            this->m_Displacements.at(i) = i * chunk;
        }
        this->m_Lengths.at(this->m_NumProcs - 1) = scrap;
    }

    static int_size getTotalNumData(const int_size numData, const MPI_Datatype& mpi_int_size)
    {
        int_size totalNumData;
        MPI_Allreduce(&numData, &totalNumData, 1, mpi_int_size, MPI_SUM, MPI_COMM_WORLD);
        return totalNumData;
    }

private:
    inline void initialize(const int_size numData) override
    {
        this->m_TotalNumData = getTotalNumData(numData, mpi_int_size);
        calcChunks();
    }
};
}  // namespace HPKmeans