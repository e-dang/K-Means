#pragma once

#include <kmeans/initializers/interface.hpp>
#include <kmeans/utils/assignment_updaters.hpp>
#include <kmeans/utils/uniform_selector.hpp>
#include <kmeans/utils/utils.hpp>
#include <kmeans/utils/weighted_selector.hpp>

namespace hpkmeans
{
constexpr char KPP[] = "k++";

template <typename T, Parallelism Level, class DistanceFunc>
class KPlusPlus : public IInitializer<T>
{
public:
    void initialize(const Matrix<T>* const data, Clusters<T>* const clusters) const override
    {
        selectFirstCluster(data, clusters);

        for (int32_t i = 1; i < clusters->maxSize(); ++i)
        {
            clusters->updateAssignments(m_updater);
            weightedClusterSelection(clusters);
        }

        clusters->updateAssignments(m_updater);
    }

private:
    void selectFirstCluster(const Matrix<T>* const data, Clusters<T>* const clusters) const
    {
        auto dataIdx = m_uniformSelector.selectSingle(data->numRows());
        clusters->addCentroid(dataIdx);
    }

    void weightedClusterSelection(Clusters<T>* const clusters) const
    {
        auto sqDistances = clusters->sqDistances();
        T randSumFrac    = getRandFraction() * std::accumulate(sqDistances->cbegin(), sqDistances->cend(), 0.0);
        auto dataIdx     = m_weightedSelector.select(sqDistances, randSumFrac);
        clusters->addCentroid(dataIdx);
    }

private:
    NewCentroidAssignmentUpdater<T, Level, DistanceFunc> m_updater;
    UniformSelector m_uniformSelector;
    WeightedSelector<T> m_weightedSelector;
};
}  // namespace hpkmeans