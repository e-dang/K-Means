#pragma once

#include <kmeans/initializers/kplusplus.hpp>
#include <memory>
#include <string>

namespace hpkmeans
{
template <typename T, Parallelism Level, class DistanceFunc>
std::unique_ptr<IInitializer<T>> createInitializer(const std::string& initializerString)
{
    if (initializerString == KPP)
        return std::make_unique<KPlusPlus<T, Level, DistanceFunc>>(new AssignmentUpdater<T, Level, DistanceFunc>());
    else if (initializerString == OPTKPP)
        return std::make_unique<KPlusPlus<T, Level, DistanceFunc>>(
          new NewCentroidAssignmentUpdater<T, Level, DistanceFunc>());
    else
        std::cerr << "Unrecognized initializer string!\n";

    exit(1);
}
}  // namespace hpkmeans