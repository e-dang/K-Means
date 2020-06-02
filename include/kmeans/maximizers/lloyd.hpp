#pragma once

#include <kmeans/maximizers/interface.hpp>
#include <kmeans/utils/assignment_updaters.hpp>

namespace hpkmeans
{
constexpr char LLOYD[]    = "lloyd";
constexpr char OPTLLOYD[] = "optlloyd";

template <typename T, Parallelism Level, class DistanceFunc>
class Lloyd : public IMaximizer<T>
{
public:
    Lloyd(AbstractAssignmentUpdater<T, DistanceFunc>* updater) : p_updater(updater) {}

    void maximize(const Matrix<T>* const data, Clusters<T>* const clusters) const override
    {
        int32_t changed;
        auto minNumChanged   = data->numRows() * this->MIN_PERCENT_CHANGED;
        auto currAssignments = clusters->assignments();
        std::vector<int32_t> prevAssignments(currAssignments->cbegin(), currAssignments->cend());

        do
        {
            clusters->template updateCentroids<Level>();

            clusters->updateAssignments(p_updater.get());

            changed = calcNumChanged(currAssignments, &prevAssignments);

        } while (changed > minNumChanged);

        clusters->template calcError<Level>();
    }

private:
    int32_t calcNumChanged(const std::vector<int32_t>* const currAssignments,
                           std::vector<int32_t>* const prevAssignments) const
    {
        int32_t numChanged = 0;

        for (int i = 0; i < static_cast<int32_t>(currAssignments->size()); ++i)
        {
            if (currAssignments->at(i) != prevAssignments->at(i))
            {
                ++numChanged;
                prevAssignments->at(i) = currAssignments->at(i);
            }
        }

        return numChanged;
    }

private:
    std::unique_ptr<AbstractAssignmentUpdater<T, DistanceFunc>> p_updater;
    // ReassignmentUpdater<T, Level, DistanceFunc> m_pointReassigner;
};
}  // namespace hpkmeans