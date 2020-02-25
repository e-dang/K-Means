#pragma once

#include <hpkmeans/data_types/cluster_results.hpp>
#include <hpkmeans/data_types/matrix.hpp>
#include <hpkmeans/distances.hpp>
#include <numeric>
#include <vector>

namespace HPKmeans
{
template <typename precision, typename int_size>
class KmeansState
{
private:
    const int m_Rank;
    const int m_NumProcs;
    const int_size m_TotalNumData;
    const std::vector<int_size> m_Lengths;
    const std::vector<int_size> m_Displacements;
    const int_size m_Displacement;

    const Matrix<precision, int_size>* const p_Data;
    const std::vector<precision>* const p_Weights;
    const std::shared_ptr<IDistanceFunctor<precision>> p_DistanceFunc;

    // dynamic data that changes each repeat
    std::shared_ptr<Matrix<precision, int_size>> p_Clusters;
    std::shared_ptr<std::vector<int_size>> p_Clustering;
    std::shared_ptr<std::vector<precision>> p_ClusterWeights;
    std::shared_ptr<std::vector<precision>> p_SqDistances;

public:
    KmeansState(const Matrix<precision, int_size>* const data, const std::vector<precision>* const weights,
                std::shared_ptr<IDistanceFunctor<precision>> distanceFunc, const int& rank, const int& numProcs,
                const int_size& totalNumData, const std::vector<int_size> lengths,
                const std::vector<int_size> displacements) noexcept :
        m_Rank(rank),
        m_NumProcs(numProcs),
        m_TotalNumData(totalNumData),
        m_Lengths(lengths),
        m_Displacements(displacements),
        m_Displacement(displacements.at(rank)),
        p_Data(data),
        p_Weights(weights),
        p_DistanceFunc(distanceFunc),
        p_Clusters(nullptr),
        p_Clustering(nullptr),
        p_ClusterWeights(nullptr),
        p_SqDistances(nullptr)
    {
    }

    ~KmeansState() = default;

    inline precision operator()(const precision* point1, const precision* point2)
    {
        return (*p_DistanceFunc)(point1, point2, dataCols());
    }

    inline void resetClusterData(const int_size& numClusters, const precision& sqDistsFillVal = -1.0)
    {
        if (!p_Clusters)
            p_Clusters = std::make_shared<Matrix<precision, int_size>>(numClusters, dataCols());
        else if (p_Clusters.use_count() == 1)
            p_Clusters->clear();

        if (!p_Clustering)
            p_Clustering = std::make_shared<std::vector<int_size>>(m_TotalNumData, -1);
        else
            std::fill(p_Clustering->begin(), p_Clustering->end(), -1);

        if (!p_ClusterWeights)
            p_ClusterWeights = std::make_shared<std::vector<precision>>(numClusters, 0.0);
        else
            std::fill(p_ClusterWeights->begin(), p_ClusterWeights->end(), 0.0);

        if (!p_SqDistances)
            p_SqDistances = std::make_shared<std::vector<precision>>(m_TotalNumData, sqDistsFillVal);
        else
            std::fill(p_SqDistances->begin(), p_SqDistances->end(), sqDistsFillVal);
    }

    inline void setClusters(std::shared_ptr<Matrix<precision, int_size>> clusters) { p_Clusters = clusters; }

    inline void compareResults(std::shared_ptr<ClusterResults<precision, int_size>> clusterResults)
    {
        precision currError = std::accumulate(p_SqDistances->begin(), p_SqDistances->end(), 0.0);

        if (clusterResults->error > currError || clusterResults->error < 0)
        {
            clusterResults->error          = currError;
            clusterResults->clusters       = std::move(p_Clusters);
            clusterResults->clustering     = std::move(p_Clustering);
            clusterResults->clusterWeights = std::move(p_ClusterWeights);
            clusterResults->sqDistances    = std::move(p_SqDistances);
        }
    }

    inline const int_size totalNumData() noexcept { return m_TotalNumData; }

    inline const int rank() noexcept { return m_Rank; }

    inline const int numProcs() noexcept { return static_cast<int>(m_Lengths.size()); }

    inline const Matrix<precision, int_size>* data() const noexcept { return p_Data; }

    inline const precision* dataAt(const int_size& idx) const { return p_Data->at(idx); }

    inline const int_size dataSize() const noexcept { return p_Data->size(); }

    inline const int_size dataCols() const noexcept { return p_Data->cols(); }

    inline const std::vector<precision>* weights() const noexcept { return p_Weights; }

    inline const precision& weightsAt(const int_size& dataIdx) const { return p_Weights->at(dataIdx); }

    inline const int_size& myLength() const { return m_Lengths.at(m_Rank); }

    inline const std::vector<int_size>& lengths() const noexcept { return m_Lengths; }

    inline const int_size& lengthsAt(const int& rank) const { return m_Lengths.at(rank); }

    inline const int_size* lengthsData() const noexcept { return m_Lengths.data(); }

    inline const int_size& myDisplacement() const noexcept { return m_Displacement; }

    inline const int_size* displacementsData() const noexcept { return m_Displacements.data(); }

    inline const std::vector<int_size>* clustering() const { return p_Clustering.get(); }

    inline int_size& clusteringAt(const int_size& dataIdx) { return p_Clustering->at(m_Displacement + dataIdx); }

    inline const int_size clusteringSize() const noexcept { return static_cast<int_size>(p_Clustering->size()); }

    inline int_size* clusteringData() noexcept { return p_Clustering->data(); }

    inline const std::vector<precision>* sqDistances() const { return p_SqDistances.get(); }

    inline precision& sqDistancesAt(const int_size& dataIdx) { return p_SqDistances->at(m_Displacement + dataIdx); }

    inline precision* sqDistancesData() noexcept { return p_SqDistances->data(); }

    inline typename std::vector<precision>::iterator sqDistancesBegin() noexcept { return p_SqDistances->begin(); }

    inline typename std::vector<precision>::iterator sqDistancesEnd() noexcept { return p_SqDistances->end(); }

    inline const std::vector<precision>* clusterWeights() const { return p_ClusterWeights.get(); }

    inline precision& clusterWeightsAt(const int_size& clusterIdx) { return p_ClusterWeights->at(clusterIdx); }

    inline const precision* clusterWeightsData() const noexcept { return p_ClusterWeights->data(); }

    inline Matrix<precision, int_size>* clusters() { return p_Clusters.get(); }

    inline const int_size clustersRows() const noexcept { return p_Clusters->rows(); }

    inline precision* clustersData() noexcept { return p_Clusters->data(); }

    inline void clustersPushBack(const precision* datapoint) { p_Clusters->push_back(datapoint); }

    inline void clustersFill(const precision& val) { p_Clusters->fill(val); }

    inline void clustersReserve(const int_size& space) { p_Clusters->reserve(space); }

    inline const int_size clustersSize() const noexcept { return p_Clusters->size(); }

    inline const int_size clustersElements() const noexcept { return p_Clusters->elements(); }

    inline const precision* clustersAt(const int_size& row) const { return p_Clusters->at(row); }
};
}  // namespace HPKmeans