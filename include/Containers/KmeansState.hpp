#pragma once

#include "Containers/DataClasses.hpp"
#include "Containers/Matrix.hpp"

namespace HPKmeans
{
template <typename precision, typename int_size>
class KmeansState
{
private:
    const int m_Rank;
    const int_size m_TotalNumData;
    const std::vector<int_size> m_Lengths;
    const std::vector<int_size> m_Displacements;
    const int_size m_Displacement;

    const Matrix<precision, int_size>* const p_Data;
    const std::vector<precision>* const p_Weights;
    const std::shared_ptr<IDistanceFunctor<precision>> p_DistanceFunc;

    // dynamic data that changes each repeat
    Matrix<precision, int_size>* p_Clusters;
    std::vector<int_size>* p_Clustering;
    std::vector<precision>* p_ClusterWeights;
    std::vector<precision>* p_SqDistances;

public:
    KmeansState(const Matrix<precision, int_size>* const data, const std::vector<precision>* const weights,
                std::shared_ptr<IDistanceFunctor<precision>> distanceFunc, const int& rank,
                const int_size& totalNumData, const std::vector<int_size> lengths,
                const std::vector<int_size> displacements) noexcept :
        m_Rank(rank),
        m_TotalNumData(totalNumData),
        m_Lengths(lengths),
        m_Displacements(displacements),
        m_Displacement(displacements.at(rank)),
        p_Data(data),
        p_Weights(weights),
        p_DistanceFunc(distanceFunc)
    {
    }

    ~KmeansState() = default;

    inline precision operator()(const precision* point1, const precision* point2, const int32_t& numFeatures)
    {
        return (*p_DistanceFunc)(point1, point2, numFeatures);
    }

    void setClusterData(ClusterData<precision, int_size>* const clusterData)
    {
        p_Clusters       = &clusterData->clusters;
        p_Clustering     = &clusterData->clustering;
        p_ClusterWeights = &clusterData->clusterWeights;
    }

    inline void setSqDistances(std::vector<precision>* const sqDistances) { p_SqDistances = sqDistances; }

    inline const int_size totalNumData() noexcept { return m_TotalNumData; }

    inline const int rank() noexcept { return m_Rank; }

    inline const Matrix<precision, int_size>* data() const noexcept { return p_Data; }

    inline const precision* dataAt(const int_size& idx) const { return p_Data->at(idx); }

    inline const int_size dataSize() const noexcept { return p_Data->size(); }

    inline const std::vector<precision>* weights() const noexcept { return p_Weights; }

    inline const precision& weightsAt(const int_size& dataIdx) const { return p_Weights->at(dataIdx); }

    inline const int_size& myLength() const { return m_Lengths.at(m_Rank); }

    inline const std::vector<int_size>& lengths() const noexcept { return m_Lengths; }

    inline const int_size* lengthsData() const noexcept { return m_Lengths.data(); }

    inline const int_size& myDisplacement() const noexcept { return m_Displacement; }

    inline const int_size* displacementsData() const noexcept { return m_Displacements.data(); }

    inline const std::vector<int_size>* clustering() const noexcept { return p_Clustering; }

    inline int_size& clusteringAt(const int_size& dataIdx) { return p_Clustering->at(m_Displacement + dataIdx); }

    inline const int_size clusteringSize() const noexcept { return static_cast<int_size>(p_Clustering->size()); }

    inline int_size* clusteringData() noexcept { return p_Clustering->data(); }

    inline const std::vector<precision>* sqDistances() const noexcept { return p_SqDistances; }

    inline precision& sqDistancesAt(const int_size& dataIdx) { return p_SqDistances->at(m_Displacement + dataIdx); }

    inline precision* sqDistancesData() noexcept { return p_SqDistances->data(); }

    inline typename std::vector<precision>::iterator sqDistancesBegin() noexcept { return p_SqDistances->begin(); }

    inline typename std::vector<precision>::iterator sqDistancesEnd() noexcept { return p_SqDistances->end(); }

    inline const std::vector<precision>* clusterWeights() const noexcept { return p_ClusterWeights; }

    inline precision& clusterWeightsAt(const int_size& clusterIdx) { return p_ClusterWeights->at(clusterIdx); }

    inline const precision* clusterWeightsData() const noexcept { return p_ClusterWeights->data(); }

    inline Matrix<precision, int_size>* clusters() noexcept { return p_Clusters; }

    inline const int_size clustersRows() const noexcept { return p_Clusters->rows(); }

    inline const int_size clustersCols() const noexcept { return p_Clusters->cols(); }

    inline precision* clustersData() noexcept { return p_Clusters->data(); }

    inline void clustersPushBack(const precision* datapoint) { p_Clusters->push_back(datapoint); }

    inline void clustersFill(const precision& val) { p_Clusters->fill(val); }

    inline void clustersReserve(const int_size& space) { p_Clusters->reserve(space); }

    inline const int_size clustersSize() const noexcept { return p_Clusters->size(); }

    inline const int_size clustersElements() const noexcept { return p_Clusters->elements(); }

    inline const precision* clustersAt(const int_size& row) const { return p_Clusters->at(row); }
};
}  // namespace HPKmeans