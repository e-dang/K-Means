#include "Factories/StrategyFactories.hpp"

IClosestClusterFinder* AbstractStrategyFactory::createClosestClusterFinder(Variant variant)
{
    switch (variant)
    {
        case (Reg):
            return new ClosestClusterFinder();
        case (Opt):
            return new ClosestNewClusterFinder();
        default:
            throw std::runtime_error("Invalid ClosestClusterFinder variant specified.");
    }
}

AbstractClosestClusterUpdater* AbstractStrategyFactory::createClosestClusterUpdater(Variant variant)
{
    switch (variant)
    {
        case (SpecificCoreset):
            return new SerialClosestClusterUpdater(createClosestClusterFinder(Reg),
                                                   createCoresetClusteringDataUpdater());
        default:
            return new SerialClosestClusterUpdater(createClosestClusterFinder(variant), createClusteringDataUpdater());
    }
}

AbstractPointReassigner* AbstractStrategyFactory::createPointReassigner(Variant variant)
{
    switch (variant)
    {
        case (Reg):
            return createRegPointReassigner();
        case (Opt):
            return createOptPointReassigner();
        default:
            throw std::runtime_error("Invalid PointReassigner variant specified.");
    }
}