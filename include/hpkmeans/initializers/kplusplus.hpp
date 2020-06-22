#pragma once

#include <mpi.h>

#include <hpkmeans/initializers/interface.hpp>
#include <hpkmeans/utils/assignment_updaters.hpp>
#include <hpkmeans/utils/uniform_selector.hpp>
#include <hpkmeans/utils/weighted_selector.hpp>

namespace hpkmeans
{
constexpr char KPP[]    = "k++";
constexpr char OPTKPP[] = "optk++";

template <typename T, Parallelism Level, class DistanceFunc>
class KPlusPlus : public IInitializer<T, Level>
{
public:
    KPlusPlus(AbstractAssignmentUpdater<T, DistanceFunc>* updater) : p_updater(updater) {}

    void initialize(const Matrix<T>* const data, Clusters<T, Level>* const clusters) const override
    {
        initializeImpl(data, clusters);
    }

private:
    template <Parallelism _Level = Level>
    std::enable_if_t<isSharedMemory(_Level)> initializeImpl(const Matrix<T>* const data,
                                                            Clusters<T, Level>* const clusters) const
    {
        selectFirstCluster(data, clusters);

        for (int32_t i = 1; i < clusters->maxSize(); ++i)
        {
            clusters->updateAssignments(p_updater.get());
            weightedClusterSelection(clusters);
        }

        clusters->updateAssignments(p_updater.get());
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level)> initializeImpl(const Matrix<T>* const data,
                                                           Clusters<T, Level>* const clusters) const
    {
        selectFirstCluster(data, clusters);

        for (int32_t i = 1; i < clusters->maxSize(); ++i)
        {
            updateAssignments(clusters);
            weightedClusterSelection(clusters);
        }

        updateAssignments(clusters);
        clusters->allGatherAssignments();
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isSharedMemory(_Level)> selectFirstCluster(const Matrix<T>* const data,
                                                                Clusters<T, Level>* const clusters) const
    {
        auto dataIdx = m_uniformSelector.selectSingle(data->numRows());
        clusters->addCentroid(dataIdx);
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isSharedMemory(_Level)> weightedClusterSelection(Clusters<T, Level>* const clusters) const
    {
        auto sqDistances = clusters->sqDistances();
        auto dataIdx = m_weightedSelector.selectSingle(sqDistances, getRandFraction() * accumulate<Level>(sqDistances));
        clusters->addCentroid(dataIdx);
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level)> selectFirstCluster(const Matrix<T>* const data,
                                                               Clusters<T, Level>* const clusters) const
    {
        int32_t dataIdx   = -1;
        int myRank        = getCommRank();
        auto totalNumData = clusters->totalNumData();

        if (myRank == 0)
            dataIdx = m_uniformSelector.selectSingle(totalNumData);

        auto rank = findRankThatHasData(dataIdx, clusters);

        distributeCentroid(rank, rank == myRank, dataIdx, clusters);
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level), int32_t> findRankThatHasData(int32_t& dataIdx,
                                                                         Clusters<T, Level>* const clusters) const
    {
        MPI_Bcast(&dataIdx, 1, MPI_INT, 0, MPI_COMM_WORLD);

        auto displacements = clusters->displacements();
        auto position      = std::lower_bound(displacements.cbegin(), displacements.cend(), dataIdx,
                                         [](const int32_t val1, const int32_t val2) { return val1 <= val2; });

        return std::distance(displacements.cbegin(), position) - 1;
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level)> distributeCentroid(const int rank, const bool hasData,
                                                               const int32_t dataIdx,
                                                               Clusters<T, Level>* const clusters) const
    {
        if (hasData)
            clusters->addCentroid(dataIdx);
        else
            clusters->reserveCentroidSpace();

        clusters->bcastCentroids(rank);
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level)> updateAssignments(Clusters<T, Level>* const clusters) const
    {
        clusters->updateAssignments(p_updater.get());
        clusters->gatherSqDistances();
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level)> weightedClusterSelection(Clusters<T, Level>* const clusters) const
    {
        int32_t dataIdx = -1;
        auto myRank     = getCommRank();

        if (myRank == 0)
        {
            auto sqDistances = clusters->sqDistances();
            dataIdx          = m_weightedSelector.selectSingle(
              sqDistances, getRandFraction() * accumulate<getConjugateParallelism<Level>()>(sqDistances));
        }

        auto rank = findRankThatHasData(dataIdx, clusters);

        distributeCentroid(rank, rank == myRank, dataIdx, clusters);
    }

private:
    std::unique_ptr<AbstractAssignmentUpdater<T, DistanceFunc>> p_updater;
    UniformSelector m_uniformSelector;
    WeightedSelector m_weightedSelector;
};
}  // namespace hpkmeans