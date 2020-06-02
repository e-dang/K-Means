#pragma once

#include <kmeans/initializers/interface.hpp>
#include <kmeans/utils/assignment_updaters.hpp>
#include <kmeans/utils/uniform_selector.hpp>
#include <kmeans/utils/weighted_selector.hpp>

namespace hpkmeans
{
constexpr char KPP[]    = "k++";
constexpr char OPTKPP[] = "optk++";

template <typename T, Parallelism Level, class DistanceFunc>
class KPlusPlus : public IInitializer<T>
{
public:
    KPlusPlus(AbstractAssignmentUpdater<T, DistanceFunc>* updater) : p_updater(updater) {}

    void initialize(const Matrix<T>* const data, Clusters<T>* const clusters) const override
    {
        selectFirstCluster(data, clusters);

        for (int32_t i = 1; i < clusters->maxSize(); ++i)
        {
            clusters->updateAssignments(p_updater.get());
            weightedClusterSelection(clusters);
        }

        clusters->updateAssignments(p_updater.get());
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
    std::unique_ptr<AbstractAssignmentUpdater<T, DistanceFunc>> p_updater;
    UniformSelector m_uniformSelector;
    WeightedSelector<T> m_weightedSelector;
};
}  // namespace hpkmeans