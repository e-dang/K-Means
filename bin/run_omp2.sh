#!/bin/bash
#$ -cwd
#$ -o ../output/omp2.txt
#$ -j y
#$ -l h_data=750M,h_rt=10:00:00
#$ -pe shared 8

. /u/local/Modules/default/init/modules.sh
module load gcc/7.2.0
module load boost/1_71_0

./kmeans -r 10000000 -c 50 -k 500 --optkpp --optlloyd --omp -f "../data/test_10000000_50.txt"