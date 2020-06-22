#pragma once

#include <kmeans/coreset/coreset_creator.hpp>
#include <kmeans/kmeans/kmeans_impl.hpp>
#include <kmeans/types/parallelism.hpp>

namespace hpkmeans
{
template <typename T, Parallelism Level, class DistanceFunc>
class CoresetKmeans : public KMeansImpl<T, Level, DistanceFunc>
{
public:
    CoresetKmeans(const std::string& initializer, const std::string& maximizer, const int repeats,
                  const int32_t sampleSize) :
        KMeansImpl<T, Level, DistanceFunc>(initializer, maximizer),
        m_coresetRepeats(repeats),
        m_sampleSize(sampleSize),
        m_coresetCreator(),
        m_assignmentUpdater(),
        m_bestCoresetClusters()
    {
    }

    const Clusters<T, Level>* const fit(const Matrix<T>* const data, const int32_t& numClusters, const int& numRepeats,
                                        const std::vector<T>* const weights = nullptr) override
    {
        for (int i = 0; i < m_coresetRepeats; ++i)
        {
            auto coreset = m_coresetCreator.createCoreset(data, m_sampleSize);

            auto coresetResults =
              KMeansImpl<T, Level, DistanceFunc>::fit(coreset.getData(), numClusters, numRepeats, coreset.getWeights());

            auto newClusters = assignNonCoresetPoints(data, coresetResults);
            newClusters.calcError();
            this->compareResults(newClusters, m_bestCoresetClusters);
        }

        return getResults();
    }

    const Clusters<T, Level>* const getResults() const override { return &m_bestCoresetClusters; }

private:
    template <Parallelism _Level = Level>
    std::enable_if_t<isSharedMemory(_Level), Clusters<T, Level>> assignNonCoresetPoints(
      const Matrix<T>* const data, const Clusters<T, Level>* const coresetResults)
    {
        Clusters<T, Level> newClusters(coresetResults->maxSize(), data);
        newClusters.copyCentroids(*coresetResults);
        newClusters.updateAssignments(&m_assignmentUpdater);
        return newClusters;
    }

    template <Parallelism _Level = Level>
    std::enable_if_t<isDistributed(_Level), Clusters<T, Level>> assignNonCoresetPoints(
      const Matrix<T>* const data, const Clusters<T, Level>* const coresetResults)
    {
        auto newClusters = assignNonCoresetPoints<getConjugateParallelism<Level>()>(data, coresetResults);
        newClusters.gatherAssignments();
        return newClusters;
    }

private:
    int m_coresetRepeats;
    int32_t m_sampleSize;
    CoresetCreator<T, Level, DistanceFunc> m_coresetCreator;
    AssignmentUpdater<T, Level, DistanceFunc> m_assignmentUpdater;
    Clusters<T, Level> m_bestCoresetClusters;
};
}  // namespace hpkmeans