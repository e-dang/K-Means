#include "Coresets.hpp"
#include "Reader.hpp"
#include "Writer.hpp"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <assert.h>

typedef boost::mt19937 RNGType;

Coresets::Coresets(int numClusters, int numRestarts, int numThreads) : numClusters(numClusters), numRestarts(numRestarts),
                                                                   numThreads(numThreads)
{

    bestError = INT_MAX;
    setNumThreads(numThreads);
}

Coresets::~Coresets()
{
}

void Coresets::initMPIMembers(int numData, int numFeatures, value_t *data)
{
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // Set data start/end indices

    // Datapoints allocated for each process to compute
    int chunk = numData / numProcs;
    int scrap = chunk + (numData % numProcs);
    // Start index for data
    startIdx_MPI = chunk * rank;
    // End index of Data
    endIdx_MPI = startIdx_MPI + chunk - 1;
    // Last process gets leftover datapoints
    if (rank == (numProcs - 1))
        endIdx_MPI = startIdx_MPI + scrap - 1;

    int size = endIdx_MPI - startIdx_MPI + 1;

    // Resize member vectors
    clusterCount_MPI.resize(numClusters);
    clusterCoord_MPI.resize(numClusters);
    for (int i = 0; i < clusterCoord_MPI.size(); i++)
    {
        clusterCoord_MPI[i].resize(numFeatures);
    }
    clustering_MPI.resize(numData);
    clusteringChunk_MPI.resize(size);

    // Size of each sub-array to gather
    vLens_MPI.resize(numProcs);
    // Index of each sub-array to gather
    vDisps_MPI.resize(numProcs);
    for (int i = 0; i < numProcs; i++)
    {
        vLens_MPI[i] = chunk;
        vDisps_MPI[i] = i * chunk;
    }
    vLens_MPI[numProcs - 1] = scrap;

    // Create disp/len arrays for data scatter
    int dataLens[numProcs];
    int dataDisps[numProcs];
    for (int i = 0; i < numProcs; i++)
    {
        dataLens[i] = vLens_MPI[i] * numFeatures;
        dataDisps[i] = vDisps_MPI[i] * numFeatures;
    }

    value_t tempData[size * numFeatures];

    // scatter data
    if (rank == 0)
    {
        assert(data != NULL);
        MPI_Scatterv(data, dataLens, dataDisps, MPI_FLOAT, tempData, dataLens[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatterv(NULL, dataLens, dataDisps, MPI_FLOAT, tempData, dataLens[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // convert to dataset_t
    data_MPI = arrayToDataset(tempData, size, numFeatures);
}

void Coresets::fit_coreset(value_t (*func)(datapoint_t &, datapoint_t &))
{
    int changed;
    int numData = coreset.data.size();
    int numFeatures = coreset.data[0].size();
    value_t currError;

    for (int run = 0; run < numRestarts; run++)
    {
        clusters = clusters_t();
        clustering = clustering_t(numData, -1);
        // initialize clusters with k++ algorithm
        kPlusPlus(coreset.data, func); 

        do
        {
            // reinitialize clusters
            for (int i = 0; i < numClusters; i++)
            {
                clusters[i] = {0, datapoint_t(numFeatures, 0.)};
            }

            // calc the weighted sum of each feature for all points belonging to a cluster
            std::vector<value_t> cluster_weights(clusters.size(), 0.0);
            for (int i = 0; i < numData; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    clusters[clustering[i]].coords[j] += coreset.weights[i] * coreset.data[i][j];
                }
                clusters[clustering[i]].count++;
                cluster_weights.at(clustering[i]) += coreset.weights[i];
            }

            // divide the sum of the points by the total cluster weight to obtain average
            for (int i = 0; i < numClusters; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    if (cluster_weights[i] < 0.000001){
                        std::cout << cluster_weights[i] << std::endl;
                    }
                    clusters[i].coords[j] /= cluster_weights[i];
                }
            }

            // reassign points to cluster
            changed = 0;
#pragma omp parallel for schedule(static), reduction(+ \
                                                                   : changed)
            for (int i = 0; i < numData; i++)
            {
                int before = clustering[i];
                nearest(coreset.data[i], i, func);
                if (before != clustering[i])
                {
                    changed++;
                }
            }
        } while (changed > (numData >> 10)); // do until 99.9% of data doesnt change

        // get total sum of distances from each point to their cluster center
        currError = 0;
        for (int i = 0; i < numData; i++)
        {
            // std::cout << currError << std::endl;
            currError += std::pow(func(coreset.data[i], clusters[clustering[i]].coords), 2);
        }

        // if this round produced lowest error, keep clustering
        if (currError < bestError)
        {
            bestError = currError;
            bestClustering = clustering;
            bestClusters = clusters;
        }
    }
}

void Coresets::kPlusPlus(dataset_t &data, value_t (*func)(datapoint_t &, datapoint_t &))
{
    RNGType rng(time(NULL));
    boost::uniform_int<> intRange(0, data.size());
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    value_t sum;
    std::vector<value_t> distances(data.size());

    // initialize first cluster randomly
    clusters.push_back({0, datapoint_t(data[intDistr()])});

    //initialize remaining clusters
    for (int clustIdx = 1; clustIdx < numClusters; clustIdx++)
    {
        // find distance between each data point and nearest cluster
        sum = 0;
#pragma omp parallel for shared(data, distances), schedule(static), reduction(+ \
                                                                              : sum)
        for (int pointIdx = 0; pointIdx < data.size(); pointIdx++)
        {
            distances[pointIdx] = nearest(data[pointIdx], pointIdx, func);
            sum += distances[pointIdx];
        }

        // select point to be next cluster center weighted by nearest distance squared
        sum *= floatDistr();
        for (int j = 0; j < data.size(); j++)
        {
            if ((sum -= distances[j]) <= 0)
            {
                clusters.push_back({0, datapoint_t(data[j])});
                break;
            }
        }
    }

// assign data points to nearest clusters
#pragma omp parallel for shared(data), schedule(static)
    for (int i = 0; i < data.size(); i++)
    {
        nearest(data[i], i, func);
    }
}

value_t Coresets::nearest(datapoint_t &point, int &pointIdx, value_t (*func)(datapoint_t &, datapoint_t &))
{
    value_t tempDist, minDist = INT_MAX - 1;

    // find distance between point and all clusters
    for (int i = 0; i < clusters.size(); i++)
    {
        tempDist = std::pow(func(point, clusters[i].coords), 2);
        if (minDist > tempDist)
        {
            minDist = tempDist;
            clustering[pointIdx] = i;
        }
    }

    return minDist;
}

void Coresets::createCoreSet(dataset_t &data, int &sampleSize, value_t (*func)(datapoint_t &, datapoint_t &))
{

    RNGType rng(time(NULL));
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);

    // calculate the mean of the data
    datapoint_t mean(data[0].size(), 0);
    auto mean_data = mean.data();

#pragma omp parallel for shared(data), schedule(static), reduction(+ : mean_data[: data[0].size()])
    for (int nth_datapoint = 0; nth_datapoint < data.size(); nth_datapoint++)
    {
        for (int i = 0; i < data[0].size(); i++)
        {
            mean_data[i] += data[nth_datapoint][i];
        }
    }

    for (int i = 0; i < mean.size(); i++)
    {
        mean[i] /= data.size();
    }

    // calculate distances between the mean and all datapoints
    double distanceSum = 0;
    std::vector<value_t> distances(data.size(), 0);
#pragma omp parallel for shared(data, distances), schedule(static), reduction(+ : distanceSum) 
    for (int i = 0; i < data.size(); i++)
    {
        distances[i] = std::pow(func(mean, data[i]), 2);
        distanceSum += distances[i];
    }

    // calculate the distribution used to choose the datapoints to create the coreset
    value_t partOne = 0.5 * (1.0 / (float)data.size()); // first portion of distribution calculation that is constant
    double sum = 0.0;
    std::vector<value_t> distribution(data.size(),0);
#pragma omp parallel for shared(distribution, distances), schedule(static), reduction(+ : sum) 
    for (int i = 0; i < data.size(); i++)
    {
        distribution[i] = partOne + 0.5 * distances[i] / distanceSum;
        // std::cout << distribution[i] << "\t";
        sum += distribution[i];
    }
    // std::cout << std::endl;
    // std::cout << "hereherehere"<< sum << std::endl;
    // create pointers to each datapoint in data
    std::vector<datapoint_t *> ptrData(data.size());
// #pragma omp parallel for shared(data, ptrData), schedule(static) // this section might have false sharing, which will degrade performance
    for (int i = 0; i < data.size(); i++)
    {
        ptrData[i] = &data[i];
    }

    // create the coreset
    double randNum;
    std::vector<datapoint_t> c(sampleSize);
    std::vector<value_t> w(sampleSize, 0);
    coreset.data = c;
    coreset.weights = w;
    for (int i = 0; i < sampleSize; i++)
    {
        randNum = floatDistr() * sum;
        for (int j = 0; j < ptrData.size(); j++)
        {
            if ((randNum -= distribution[j]) <= 0)
            {
                coreset.data[i] = *(ptrData[j]);
                // coreset.data[i] = datapoint_t(ptrData[j], ptrData[j] + 2);
                // for (int k = 0; k < coreset.data[i].size(); k++){
                //     std::cout <<  coreset.data[i][k];
                // }
                // std::cout << std::endl;
                // std::copy(coreset.data[i], ptrData[j], ptrData[j] + numFeatures);
                coreset.weights[i] = (double) (1.0 / (sampleSize * distribution[j]));

                // std::cout << coreset.weights[i] << std::endl;

                sum -= distribution[j];
                ptrData.erase(ptrData.begin() + j);
                distribution.erase(distribution.begin() + j);
                break;
            }
        }
    }
}

void Coresets::createCoreSet_MPI(int numData, int numFeatures, value_t *data, int sampleSize, value_t (*func)(datapoint_t &, datapoint_t &)){
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // scatter the data across all processes
    initMPIMembers(numData, numFeatures, data);

    RNGType rng(time(NULL));
    boost::uniform_int<> intRange(0, data_MPI.size());
    boost::uniform_real<> floatRange(0, 1);
    boost::variate_generator<RNGType, boost::uniform_int<>> intDistr(rng, intRange);
    boost::variate_generator<RNGType, boost::uniform_real<>> floatDistr(rng, floatRange);
    
    // compute the mean, sum, and sqd sum of the data local to each machine
    datapoint_t local_mean(data_MPI[0].size(), 0);
    datapoint_t local_sum(data_MPI[0].size(), 0);
    datapoint_t local_sqd_sum(data_MPI[0].size(), 0);

    auto local_mean_data = local_mean.data();
    auto local_sum_data = local_sum.data();
    auto local_sqd_sum_data = local_sqd_sum.data();

    for (int nth_datapoint = 0; nth_datapoint < data_MPI.size(); nth_datapoint++)
    {
        // std::cout << rank << " rank's datapoint: "; 
        for (int mth_feature = 0; mth_feature < data_MPI[0].size(); mth_feature++)
        {
            local_mean_data[mth_feature] += data_MPI[nth_datapoint][mth_feature];
            local_sum_data[mth_feature] +=  data_MPI[nth_datapoint][mth_feature];
            local_sqd_sum_data[mth_feature] += std::pow(data_MPI[nth_datapoint][mth_feature], 2);
            // std::cout<< data_MPI[nth_datapoint][mth_feature] << "\t";
        }
    }
    // std::cout << local_sum[0] << "\t" << local_sqd_sum[0] <<std::endl;

    for (int i = 0; i < local_mean.size(); i++)
    {
        local_mean[i] /= data_MPI.size();
        // std::cout << local_mean[i] << std::endl;
    }
    int local_cardinality = data_MPI.size();

    // gather the local sums, squared sums, and means onto central machine
    float *mean = NULL;
    float *sum = NULL;
    float *sqd_sum = NULL;
    int *local_cardinalities = NULL;
    if (rank == 0){
        mean = (float*)malloc(numProcs*data_MPI[0].size()*sizeof(float));
        sum = (float*)malloc(numProcs*data_MPI[0].size()*sizeof(float));
        sqd_sum =(float*) malloc(numProcs*data_MPI[0].size()*sizeof(float));
        local_cardinalities = (int*)malloc(numProcs*sizeof(int));
    }

    MPI_Gather(local_mean.data(), data_MPI[0].size(), MPI_FLOAT, mean, data_MPI[0].size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_sum.data(), data_MPI[0].size(), MPI_FLOAT, sum, data_MPI[0].size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(local_sqd_sum.data(), data_MPI[0].size(), MPI_FLOAT, sqd_sum, data_MPI[0].size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_cardinality, 1, MPI_INT, local_cardinalities, 1, MPI_INT, 0, MPI_COMM_WORLD);


    // root machine will compute the following:
    std::vector<value_t> global_mean(data_MPI[0].size(),0);
    std::vector<value_t> quant_errs(numProcs, 0);
    std::vector<int> uniform_sample_counts(numProcs,0); // number of points to be sampled uniformly on the ith machine
    std::vector<int> phi_sample_counts(numProcs,0); // number of points to be sampled based on quant error on the ith machine
    std::vector<int> samples_per_proc(numProcs, 0); // Elements represent number of data points to be sampled per machine. Index corresponds to rank of the proccess. 
    std::vector<int> samples_per_proc_disp(numProcs, 0);
    std::vector<int> data_per_proc(numProcs, 0); // Elements represent the amount of data (number of floating point numbers) to sample on each machine (element-wise sum of the above two vectors)*dataset dimensionality
    std::vector<int> data_per_proc_disp(numProcs, 0);
    int dataset_cardinality = 0;
    float total_quant_err = 0.0;

    if (rank == 0){
        // compute the global mean from the gathered mean data
        std::vector<datapoint_t> local_means;
        for (int nth_proc = 0; nth_proc <numProcs; nth_proc++){
            for (int mth_feature = 0; mth_feature < data_MPI[0].size(); mth_feature++){
                global_mean[mth_feature] += mean[(nth_proc*data_MPI[0].size()) + mth_feature];
                // std::cout << mean[(nth_proc*data_MPI[0].size()) + mth_feature] << std::endl;
            }
            std::vector<value_t> loc_mean(&mean[nth_proc*data_MPI[0].size()], &mean[nth_proc*data_MPI[0].size() + data_MPI[0].size()]);
            local_means.push_back(loc_mean);
        }
        // std::cout << "here" << std::endl;
        for (int mth_feature = 0; mth_feature < data_MPI[0].size(); mth_feature++){
            global_mean[mth_feature] /= (float)numProcs;
            // std::cout << global_mean[mth_feature] << std::endl;
        }

        // compute the total cardinality of the dataset -- might be unecessary but doing this to make this section of code robust to chagne in the way we get a dataset
        for (int nth_proc = 0; nth_proc < numProcs; nth_proc++){
            dataset_cardinality += local_cardinalities[nth_proc];
            // std::cout << "local_cardinalities: " << local_cardinalities[nth_proc] << std::endl;
        }
        // std::cout << "dataset cardinality: "<< dataset_cardinality << std::endl;

        // compute the local quantization error for each machine and the total quantization error
        for (int nth_proc = 0; nth_proc < numProcs; nth_proc++){
            value_t ith_quant_err;
            // for (int mth_feature = 0; mth_feature < data_MPI[0].size(); mth_feature++){
            //     ith_quant_err += sqd_sum[(nth_proc*data_MPI[0].size())+mth_feature] - 2*global_mean[mth_feature]*sum[(nth_proc*data_MPI[0].size())+mth_feature] + local_cardinalities[nth_proc]*std::pow(global_mean[mth_feature],2);
            // }
            // std::cout << "here2"<<local_means.size()<< std::endl; 

            ith_quant_err = std::pow(func(global_mean, local_means[nth_proc]),2);
            // std::cout << "here3"<< std::endl; 
            quant_errs[nth_proc] = ith_quant_err;
            // std::cout<< "ith_quant_err: " << ith_quant_err << std::endl;
            total_quant_err += ith_quant_err;
        }
        // std::cout << "total_quant_err: " << total_quant_err << std::endl;

        // compute the number of points to sample from each machine
        double randNum;
        for (int nth_sample = 0; nth_sample < sampleSize; nth_sample ++){
            randNum = floatDistr();
            if (randNum > .5){
                randNum = floatDistr()*dataset_cardinality;
                int cumsum = 0;
                for (int nth_proc = 0; nth_proc < numProcs; nth_proc++){
                    cumsum += local_cardinalities[nth_proc];
                    if (cumsum >= randNum) {
                        uniform_sample_counts[nth_proc] += 1;
                        samples_per_proc[nth_proc] += 1;
                        data_per_proc[nth_proc] += data_MPI[0].size();
                        break;
                    }
                }
            }
            else {
                randNum = floatDistr()*total_quant_err;
                float cumsum = 0;
                for (int nth_proc = 0; nth_proc < numProcs; nth_proc++){
                    cumsum += quant_errs[nth_proc];
                    if (cumsum >= randNum) {
                        phi_sample_counts[nth_proc] += 1;
                        samples_per_proc[nth_proc] += 1;
                        data_per_proc[nth_proc] += data_MPI[0].size();
                        break;
                    }
                }
            }
        }

        // calculate the displacement of the coreset data indices in the soon-to-be aggregated coreset array
        // std::cout << "SAMPLES PER PROC PHI: " << phi_sample_counts[0] << " "<< phi_sample_counts[1] << std::endl;
        // std::cout << "SAMPLES PER PROC UNIFORM: " << uniform_sample_counts[0] << " "<< uniform_sample_counts[1] << std::endl;
        // std::cout << "SAMPLES PER PROC: " << samples_per_proc[0] << std::endl;
        // std::cout << "SAMPLES PER PROC: " << samples_per_proc[1] << std::endl;
        for (int nth_proc = 1; nth_proc < numProcs; nth_proc++){
            data_per_proc_disp[nth_proc] = data_per_proc[nth_proc-1] + data_per_proc_disp[nth_proc];
            samples_per_proc_disp[nth_proc] = samples_per_proc[nth_proc-1] + samples_per_proc_disp[nth_proc];
        }
    }   
    // need barrier here?
    // MPI_Barrier(MPI_COMM_WORLD);
    // broadcast the global mean, total quantization error, sampling counts and local quantization errors
    MPI_Bcast(global_mean.data(), data_MPI[0].size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_quant_err, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dataset_cardinality, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(quant_errs.data(), numProcs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(phi_sample_counts.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(uniform_sample_counts.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);

    // calculate the distribution used to choose the datapoints to create the coreset
    // std::cout << "process number: " << rank <<" has this size "<< data_MPI.size() << std::endl;
    value_t partOne = 0.5 * (1.0 / dataset_cardinality); // first portion of distribution calculation that is constant
    std::vector<value_t> distribution(data_MPI.size(),0);
    std::vector<value_t> sqd_distances(data_MPI.size(),0);
    float s = 0;
    for (int i = 0; i < data_MPI.size(); i++)
    {
        sqd_distances[i] = std::pow(func(data_MPI[i], global_mean), 2);
        distribution[i] = partOne + .5*sqd_distances[i]/(dataset_cardinality*total_quant_err);
        // std::cout << "asdf" << func(data_MPI[i], global_mean) << std::endl;
        // std::cout << distribution[i] << std::endl;
        s += distribution[i];
    }
    // std::cout << s << std::endl;

    // create pointers to each datapoint in data
    std::vector<datapoint_t *> ptrData(data_MPI.size());
    for (int i = 0; i < data_MPI.size(); i++)
    {
        ptrData[i] = &data_MPI[i];
    }
    // std::cout << "helloooooo" << std::endl;
    // generate the coreset by first sampling the appropriate number of points for a given machine from the uniform distribution
    int randNum;
    double phiDist = total_quant_err; 
    double uniformDist = data_MPI.size() -1; 
    int nth_uniform_sample;
    std::vector<datapoint_t> c(uniform_sample_counts[rank] + phi_sample_counts[rank]);
    std::vector<value_t> w(uniform_sample_counts[rank] + phi_sample_counts[rank], 0);
    for (nth_uniform_sample = 0; nth_uniform_sample < uniform_sample_counts[rank]; nth_uniform_sample++){
        randNum = (int) std::round(floatDistr()*(uniformDist - nth_uniform_sample)); // subtract the contribution of the previously sampled point from the uniform dist
        // if (randNum > distribution.size()){
            // std::cout << "RandNum: " << randNum << " Size: " << distribution.size() << std::endl;
        // }
        c[nth_uniform_sample] = *ptrData[randNum];
        w[nth_uniform_sample] = (double) (1.0 / (sampleSize * distribution[randNum]));
        // std::cout << sampleSize * distribution[randNum] << std::endl;
        // std::cout <<nth_uniform_sample<< " "<< w[nth_uniform_sample] << std::endl;
        if (w[nth_uniform_sample] < .000001){
            std::cout << w[nth_uniform_sample] << std::endl;
        }
        phiDist -= sqd_distances[nth_uniform_sample]; // subtract contribution of the sampled point from the phi distrubition
        ptrData.erase(ptrData.begin() + randNum);
        distribution.erase(distribution.begin() + randNum);
    }
    // std::cout <<rank << "hello8  " << std::endl;
    double randPhi;
    for (int i = uniform_sample_counts[rank]; i < uniform_sample_counts[rank] + phi_sample_counts[rank]; i++)
    {
        randPhi = floatDistr() * phiDist;
        for (int j = 0; j < ptrData.size(); j++)
        {
            if ((randPhi -= sqd_distances[i]) <= 0)
            {
                c[i] = *ptrData[j];
                w[i] = (double) (1.0 / (sampleSize * distribution[j]));
                if (w[i] < .000001){
                    std::cout << w[i] << std::endl;
                }
                // std::cout << i << " "<< w[i] << std::endl;
                // for (int j = 0; j < data_MPI[0].size(); j++){
                //     std::cout<< c[i][j] << "\t";
                // }
                // std::cout << std::endl;
                phiDist -= sqd_distances[i];
                ptrData.erase(ptrData.begin() + j);
                distribution.erase(distribution.begin() + j);
                break;
            }
        }
    }
    // std::cout<< rank<< "coreset cardinality: " << c.size() << std::endl;
    // for (int i = 0; i < c.size(); i++){
    //     for (int j = 0; j < data_MPI[0].size(); j++){
    //         std::cout << c[i][j] << "\t";
    //     }
    //     std::cout << std::endl;
    // }
    // for (int i = 0; i < w.size(); i++){
    //     std::cout << rank << " " << w[i] << std::endl;
    // }
    // flatten the 2d vector of coreset data
    // float* flattenedCoresetData = (float*) malloc(sampleSize*data_MPI[0].size()*sizeof(float));
    int local_coreset_cardinality = c.size();
    std::vector<value_t> flattenedCoresetData(local_coreset_cardinality*data_MPI[0].size());
    // std::cout <<rank<<"flattened array size: "<< local_coreset_cardinality*data_MPI[0].size() << std::endl;

    for (int i = 0; i < local_coreset_cardinality; i++){
            // std::cout <<"i: "<< i << " dim: " << dim << std::endl;
        for (int j = 0; j < data_MPI[0].size(); j++){
            // std::cout << i*data_MPI.size() + j << std::endl;
            flattenedCoresetData[i*data_MPI[0].size() + j] = c[i][j];

        // for (int j = 0; j < data_MPI[0].size(); j++){
        //         std::cout<< flattenedCoresetData[i*data_MPI[0].size() + j] << "\t";
        // }
        // std::cout << std::endl;
        }
    }
    // std::cout <<rank << "hello10" << std::endl;

    //gather the coresets back onto the root machine
    value_t* coreset_temp = NULL;
    value_t* weights_temp = NULL;

    if (rank == 0) {
        coreset_temp = (value_t*)malloc(sampleSize*data_MPI[0].size()*sizeof(float));
        weights_temp = (value_t*)malloc(sampleSize*sizeof(float));
    }
    // std::cout <<rank << "hello10" << std::endl;

    int data_send_count = data_MPI[0].size()*local_coreset_cardinality;
    // std::cout <<rank <<"data send count: "<< data_send_count << std::endl;
    // std::cout <<rank <<"recv buffer size: "<< sampleSize*data_MPI[0].size() << std::endl;
    // std::cout <<rank <<"elements reserved for data from rank 0: "<< data_per_proc[0] << std::endl;
    // std::cout <<rank <<"elements reserved for data from rank 1: "<< data_per_proc[1] << std::endl;
    // std::cout <<rank <<"displacement for data from rank 0: "<< data_per_proc_disp[0] << std::endl;
    // std::cout <<rank <<"displacement for data from rank 1: "<< data_per_proc_disp[1] << std::endl;


    MPI_Gatherv(flattenedCoresetData.data(), data_send_count, MPI_FLOAT, coreset_temp, data_per_proc.data(), data_per_proc_disp.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    // std::cout <<rank << "hello11" << std::endl;

    MPI_Gatherv(w.data(), local_coreset_cardinality, MPI_FLOAT, weights_temp, samples_per_proc.data(), samples_per_proc_disp.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    // std::cout <<rank << "hello12" << std::endl;

    if (rank == 0){
        // organize the flattened coresets_temp and weights_temp arrays into a coresets_t struct
        std::vector<datapoint_t> coreset_data;
        for (int i = 0; i < sampleSize; i ++){
            datapoint_t datapoint_temp(coreset_temp + i*data_MPI[0].size(), coreset_temp+ (i+1)*data_MPI[0].size());
            coreset_data.push_back(datapoint_temp);
            // for (int j = 0; j < data_MPI[0].size(); j++){
            //     std::cout<< coreset_temp[i*data_MPI[0].size() + j] << "\t";
            // }
            // std::cout << std::endl;
        }
        coreset.data = coreset_data;

        std::vector<float> w(weights_temp, weights_temp+sampleSize);
        coreset.weights = w;
        // for (int i = 0; i < sampleSize; i++){
        //     std::cout << weights_temp[i] << std::endl;
        // }
    }
}

dataset_t Coresets::arrayToDataset(value_t *data, int size, int numFeatures)
{

    dataset_t dataVec = dataset_t(size, datapoint_t(numFeatures));

    for (int i = 0; i < dataVec.size(); i++)
    {
        for (int j = 0; j < numFeatures; j++)
        {
            dataVec[i][j] = data[(i * numFeatures) + j];
        }
    }
    return dataVec;
}

datapoint_t Coresets::getClusterCoord(int idx, int numFeatures)
{
    value_t coord[numFeatures];
    MPI_Get(coord, numFeatures, MPI_FLOAT, 0, idx * numFeatures, numFeatures, MPI_FLOAT, clusterCoordWin);

    datapoint_t coordVec(numFeatures);

    for (int i = 0; i < coordVec.size(); i++)
    {
        coordVec[i] = coord[i];
    }
    return coordVec;
}
int Coresets::getClusterCount(int idx)
{
    int count;
    MPI_Get(&count, 1, MPI_INT, 0, idx, 1, MPI_INT, clusterCoordWin);
    return count;
}

int Coresets::getClustering(int idx)
{
    int count;
    MPI_Get(&count, 1, MPI_INT, 0, idx, 1, MPI_INT, clusteringWin);
    return count;
}

void Coresets::setClusterCount(int idx, int *count)
{
    MPI_Put(count, 1, MPI_INT, 0, idx, 1, MPI_INT, clusterCountWin);
}

void Coresets::setClusterCoord(int idx, int numFeatures, datapoint_t *coord)
{
    MPI_Put(coord, coord->size(), MPI_FLOAT, 0, idx * numFeatures, coord->size(), MPI_FLOAT, clusterCoordWin);
}

bool Coresets::setNumClusters(int numClusters)
{
    this->numClusters = numClusters;
    return true;
}

bool Coresets::setNumRestarts(int numRestarts)
{
    this->numRestarts = numRestarts;
    return true;
}

bool Coresets::setNumThreads(int numThreads)
{
    this->numThreads = numThreads;
    omp_set_num_threads(this->numThreads);
    return true;
}

value_t Coresets::distanceL2(datapoint_t &p1, datapoint_t &p2)
{
    value_t sum = 0;
    for (int i = 0; i < p1.size(); i++)
    {
        sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }

    return std::sqrt(sum);
}