#pragma once

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"

class IKmeansDataCreator
{
public:
    virtual KmeansData create(const Matrix* const data, const std::vector<value_t>* const weights,
                              std::shared_ptr<IDistanceFunctor> distanceFunc) = 0;
};

class SharedMemoryKmeansDataCreator : public IKmeansDataCreator
{
public:
    KmeansData create(const Matrix* const data, const std::vector<value_t>* const weights,
                      std::shared_ptr<IDistanceFunctor> distanceFunc) override;
};

class MPIKmeansDataCreator : public IKmeansDataCreator
{
public:
    KmeansData create(const Matrix* const data, const std::vector<value_t>* const weights,
                      std::shared_ptr<IDistanceFunctor> distanceFunc) override;
};