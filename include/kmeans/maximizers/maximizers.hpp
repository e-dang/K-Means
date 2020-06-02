#pragma once

#include <kmeans/maximizers/lloyd.hpp>
#include <memory>
#include <string>

namespace hpkmeans
{
template <typename T, Parallelism Level, class DistanceFunc>
std::unique_ptr<IMaximizer<T>> createMaximizer(const std::string& maximizerString)
{
    if (maximizerString == LLOYD)
        return std::make_unique<Lloyd<T, Level, DistanceFunc>>(new AssignmentUpdater<T, Level, DistanceFunc>());
    else if (maximizerString == OPTLLOYD)
        return std::make_unique<Lloyd<T, Level, DistanceFunc>>(new ReassignmentUpdater<T, Level, DistanceFunc>());
    else
        std::cerr << "Unrecognized maximizer string!\n";

    exit(1);
}
}  // namespace hpkmeans