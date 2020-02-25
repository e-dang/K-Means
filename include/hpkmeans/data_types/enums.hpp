#pragma once

namespace HPKmeans
{
enum Initializer
{
    InitNull = 0,
    KPP      = 1 << 0,
    OptKPP   = 1 << 1
};

enum Maximizer
{
    MaxNull  = 0,
    Lloyd    = 1 << 2,
    OptLloyd = 1 << 3
};

enum CoresetCreator
{
    None      = 0,
    LWCoreset = 1 << 4,
};

enum Parallelism
{
    ParaNull = 0,
    Serial   = 1 << 5,
    OMP      = 1 << 6,
    MPI      = 1 << 7,
    Hybrid   = 1 << 8
};

enum Variant
{
    Reg,
    Opt,
    SpecificCoreset
};
}  // namespace HPKmeans