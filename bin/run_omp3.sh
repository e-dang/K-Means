#!/bin/bash
#$ -cwd
#$ -o ../output/omp3.txt
#$ -j y
#$ -l h_data=750M,h_rt=10:00:00
#$ -pe shared 8

. /u/local/Modules/default/init/modules.sh
module load gcc/7.2.0
module load boost/1_71_0

OMP_NUM_THREADS=8

./kmeans -r 10000000 -c 50 -k 500 -s 1000000 --kpp --lloyd --coreset --omp -f "../data/test_10000000_50.txt"