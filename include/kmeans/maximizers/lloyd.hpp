#pragma once

#include <kmeans/maximizers/interface.hpp>
#include <kmeans/utils/assignment_updaters.hpp>
#include <kmeans/utils/centroid_updater.hpp>
namespace hpkmeans
{
constexpr char LLOYD[]    = "lloyd";
constexpr char OPTLLOYD[] = "optlloyd";

template <Parallelism Level>
std::enable_if_t<isSingleThreaded(Level), int32_t> calcNumChanged(const VectorView<int32_t>* const currAssignments,
                                                                  VectorView<int32_t>* const prevAssignments)
{
    int32_t numChanged = 0;

    for (int i = 0; i < currAssignments->viewSize(); ++i)
    {
        if (currAssignments->at(i) != prevAssignments->at(i))
        {
            ++numChanged;
            prevAssignments->at(i) = currAssignments->at(i);
        }
    }

    return numChanged;
}

template <Parallelism Level>
std::enable_if_t<isMultiThreaded(Level), int32_t> calcNumChanged(const VectorView<int32_t>* const currAssignments,
                                                                 VectorView<int32_t>* const prevAssignments)
{
    int32_t numChanged = 0;

#pragma omp parallel for schedule(static), reduction(+ : numChanged)
    for (int i = 0; i < static_cast<int32_t>(currAssignments->viewSize()); ++i)
    {
        if (currAssignments->at(i) != prevAssignments->at(i))
        {
            ++numChanged;
            prevAssignments->at(i) = currAssignments->at(i);
        }
    }

    return numChanged;
}

template <typename T, Parallelism Level, class DistanceFunc>
class Lloyd : public IMaximizer<T, Level>
{
public:
    Lloyd(AbstractAssignmentUpdater<T, DistanceFunc>* updater) : p_assignmentUpdater(updater) {}

    void maximize(const Matrix<T>* const data, Clusters<T, Level>* const clusters) const override
    {
        int32_t changed;
        auto minNumChanged   = data->numRows() * this->MIN_PERCENT_CHANGED;
        auto currAssignments = clusters->assignments();
        VectorView<int32_t> prevAssignments(*currAssignments);

        do
        {
            changed = maximizeIter(clusters, currAssignments, &prevAssignments);

        } while (changed > minNumChanged);

        clusters->calcError();
    }

private:
    template <Parallelism _Level = Level>
    std::enable_if_t<isSharedMemory(_Level), int32_t> maximizeIter(Clusters<T, Level>* const clusters,
                                                                   const VectorView<int32_t>* const currAssignments,
                                                                   VectorView<int32_t>* const prevAssignments) const
    {
        clusters->updateCentroids(m_centroidUpdater);
        clusters->updateAssignments(p_assignmentUpdater.get());
        auto changed = calcNumChanged<Level>(currAssignments, prevAssignments);

        return changed;
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level), int32_t> maximizeIter(Clusters<T, Level>* const clusters,
                                                                  const VectorView<int32_t>* const currAssignments,
                                                                  VectorView<int32_t>* const prevAssignments) const
    {
        auto changed = maximizeIter<getConjugateParallelism<Level>()>(clusters, currAssignments, prevAssignments);

        MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        clusters->allGatherAssignments();

        return changed;
    }

private:
    std::unique_ptr<AbstractAssignmentUpdater<T, DistanceFunc>> p_assignmentUpdater;
    CentroidUpdater<T, Level> m_centroidUpdater;
};
}  // namespace hpkmeans