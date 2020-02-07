#pragma once

#include "DataClasses.hpp"
#include "Definitions.hpp"
#include "Utils.hpp"

class IKmeansDataCreator
{
public:
    virtual KmeansData create(Matrix* data, std::vector<value_t>* weights,
                              std::shared_ptr<IDistanceFunctor> distanceFunc) = 0;
};

class SharedMemoryKmeansDataCreator : public IKmeansDataCreator
{
public:
    KmeansData create(Matrix* data, std::vector<value_t>* weights,
                      std::shared_ptr<IDistanceFunctor> distanceFunc) override
    {
        return KmeansData(data, weights, distanceFunc, 0, data->getNumData(),
                          std::vector<int_fast32_t>(1, data->getNumData()), std::vector<int_fast32_t>(1, 0));
    }
};

class MPIKmeansDataCreator : public IKmeansDataCreator
{
public:
    KmeansData create(Matrix* data, std::vector<value_t>* weights,
                      std::shared_ptr<IDistanceFunctor> distanceFunc) override
    {
        auto totalNumData = getTotalNumDataMPI(data);
        auto mpiData      = getMPIData(totalNumData);
        return KmeansData(data, weights, distanceFunc, mpiData.rank, totalNumData, mpiData.lengths,
                          mpiData.displacements);
    }
};