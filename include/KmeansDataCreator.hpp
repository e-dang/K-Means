#pragma once

#include "DataClasses.hpp"
#include "Definitions.hpp"
#include "Utils.hpp"

class IKmeansDataCreator
{
public:
    virtual KmeansData create(Matrix* data, std::vector<value_t>* weights,
                              std::shared_ptr<IDistanceFunctor> distanceFunc) = 0;

    virtual unsigned long getTotalNumData(const Matrix* const data) = 0;
};

class SharedMemoryKmeansDataCreator : public IKmeansDataCreator
{
public:
    KmeansData create(Matrix* data, std::vector<value_t>* weights,
                      std::shared_ptr<IDistanceFunctor> distanceFunc) override
    {
        return KmeansData(data, weights, distanceFunc, 0, data->getNumData(), std::vector<int>(1, data->getNumData()),
                          std::vector<int>(1, 0));
    }

    virtual unsigned long getTotalNumData(const Matrix* const data) { return data->getNumData(); }
};

class MPIKmeansDataCreator : public IKmeansDataCreator
{
public:
    KmeansData create(Matrix* data, std::vector<value_t>* weights,
                      std::shared_ptr<IDistanceFunctor> distanceFunc) override
    {
        auto totalNumData = getTotalNumData(data);
        auto mpiData      = getMPIData(totalNumData);
        return KmeansData(data, weights, distanceFunc, mpiData.rank, totalNumData, mpiData.lengths,
                          mpiData.displacements);
    }

    virtual unsigned long getTotalNumData(const Matrix* const data) { return getTotalNumDataMPI(data); }
};