#!/bin/bash
#$ -cwd
#$ -o ../output/hybrid2.txt
#$ -j y
#$ -l h_data=250M,h_rt=10:00:00
#$ -pe 8threads* 3

. /u/local/Modules/default/init/modules.sh
module load gcc/7.2.0
module load openmpi/3.0.0
module load boost/1_71_0

OMP_NUM_THREADS=8

mpirun ./kmeans -r 10000000 -c 50 -k 500 --optkpp --optlloyd --hybrid -f "../data/test_10000000_50.txt"