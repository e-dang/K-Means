#pragma once

#include "Containers/DataClasses.hpp"
#include "Containers/Definitions.hpp"
#include "Strategies/ClosestClusterUpdater.hpp"

class AbstractPointReassigner
{
protected:
    std::unique_ptr<AbstractClosestClusterUpdater> pUpdater;

public:
    AbstractPointReassigner(AbstractClosestClusterUpdater* updater) : pUpdater(updater){};

    virtual ~AbstractPointReassigner(){};

    int32_t reassignPoint(const int32_t& dataIdx, KmeansData* const kmeansData);

    virtual int32_t reassignPoints(KmeansData* const kmeansData) = 0;
};

class SerialPointReassigner : public AbstractPointReassigner
{
public:
    SerialPointReassigner(AbstractClosestClusterUpdater* updater) : AbstractPointReassigner(updater) {}

    ~SerialPointReassigner(){};

    int32_t reassignPoints(KmeansData* const kmeansData) override;
};

class SerialOptimizedPointReassigner : public AbstractPointReassigner
{
public:
    SerialOptimizedPointReassigner(AbstractClosestClusterUpdater* updater) : AbstractPointReassigner(updater) {}

    ~SerialOptimizedPointReassigner(){};

    int32_t reassignPoints(KmeansData* const kmeansData) override;
};

class OMPPointReassigner : public AbstractPointReassigner
{
public:
    OMPPointReassigner(AbstractClosestClusterUpdater* updater) : AbstractPointReassigner(updater) {}

    ~OMPPointReassigner(){};

    int32_t reassignPoints(KmeansData* const kmeansData) override;
};

class OMPOptimizedPointReassigner : public AbstractPointReassigner
{
public:
    OMPOptimizedPointReassigner(AbstractClosestClusterUpdater* updater) : AbstractPointReassigner(updater) {}

    ~OMPOptimizedPointReassigner(){};

    int32_t reassignPoints(KmeansData* const kmeansData) override;
};