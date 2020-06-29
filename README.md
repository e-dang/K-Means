# K-Means

## Description
K-Means library written in C++17 using the K-Means algorithms K++<sup>[1]</sup> and Lloyd's algorithm<sup>[2]</sup>. In addition, this library also includes an implementation of a K-Means approximation algorithm using lightweight coresets<sup>[3]</sup>, as well as an optimization to Lloyd's algorithm<sup>[4]</sup>. Each algorithm has been implemented for serial, muli-threaded, distributed, and hybrid execution. The level of parallelism, as well as the distance function that is used for clustering, is selected statically at compile time. Benchmarks for the optimized Lloyd and lightweight coreset algorithms, which ran on a centOS cluster using a dataset consisting of 1,000,000 datapoints each with dimensionality of 10, organized into 50 clusters, where each cluster was generated from a guassian distribution with standard deviation of 1 in a (-100, 100) box, can be seen below. For each run, the KMeans algorithm was repeated 10 times. In runs that used the lightweight coresets algorithm<sup>[3]</sup>, the sample size was 1/10<sup>th</sup> the size of full dataset. The scripts used to generate the test data and plot the results can be found in the __results__ directory.

<div style="text-align:center">
<img src="results/plots/omp_optlloyd.png" width="300"/>
<img src="results/plots/omp_optlloyd_coreset.png" width="300"/>
</div>
<div style="text-align:center">
<img src="results/plots/mpi_optlloyd.png" width="300"/>
<img src="results/plots/mpi_optlloyd_coreset.png" width="300"/>
</div>
<div style="text-align:center">
<img src="results/plots/hybrid_optlloyd.png" width="300"/>
<img src="results/plots/hybrid_optlloyd_coreset.png" width="300"/>
</div>

## Dependencies
- C++17
- Boost 1.72.0
- OpenMPI 4.0.2
- OpenMP compatible C++ compiler
- [Matrix](https://github.com/e-dang/Matrix.git)

## Compiling
Tested on MacOS Mojave 10.14.6 and CentOS with compilers:
- Clang 9.0.1
- GCC 7.2.0

In the top level directory of the project, run the following commands:
```
mkdir build
cd build
cmake ..
make
```

## Usage
This library may be imported into an existing C++ project in which case you need to compile and link the hpkmeans library to your project, then add the include statement to the top level __kmedoids.hpp__ header file to start using it.
```
#include <hpkmeans/kmeans.hpp>
```

There is also a __main.cpp__ src file included which can be edited to specify a filepath to a data file, the dimensions of the data, the hyperparameters, the parallelism, and the algorithms to use for clustering. Compiling and running this program will produce result files containing the centroids, assignments, and error in the same directory as the data file.

## Authors
All code in the current version was written by:
- Eric Dang

[Original version written for our class](https://docs.google.com/document/d/16eGwxOLUhvTCcHL0FJ_clUwuB37_pZPFDyxGHFDVrWA/edit) (all code written prior to 2020) was written by:
- Eric Dang
- [Trevor Rollins](https://github.com/tkrollins/K-Means)
- [Will Yang](https://github.com/pr33ch/K-Means)

## References
1. Arthur, D.; Vassilvitskii, S. (2007). "[k-means++: the advantages of careful seeding](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)" (PDF). Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms. Society for Industrial and Applied Mathematics Philadelphia, PA, USA. pp. 1027–1035.
2.  Lloyd, Stuart P. (1982), "[Least squares quantization in PCM](https://cs.nyu.edu/~roweis/csc2515-2006/readings/lloyd57.pdf)" (PDF), IEEE Transactions on Information Theory, 28 (2): 129–137, doi:10.1109/TIT.1982.1056489.
3.  Olivier Bachem, Mario Lucic, and Andreas Krause. 2018. [Scalable k-Means
Clustering via Lightweight Coresets](https://arxiv.org/pdf/1702.08248.pdf). In KDD ’18: The 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, August
19–23, 2018, London, United Kingdom. ACM, New York, NY, USA, 9 pages.
https://doi.org/10.1145/3219819.3219973
4. Fahim, A. M., et al. [An efficient enhanced k-means clustering algorithm](https://link.springer.com/content/pdf/10.1631/jzus.2006.A1626.pdf). Journal of Zhejiang University-Science A 7.10 (2006): 1626-1633.