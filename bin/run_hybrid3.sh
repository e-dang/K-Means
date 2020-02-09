#!/bin/bash
#$ -cwd
#$ -o ../output/hybrid3.txt
#$ -j y
#$ -l h_data=250M,h_rt=10:00:00
#$ -pe 8threads* 3

. /u/local/Modules/default/init/modules.sh
module load gcc/7.2.0
module load openmpi/3.0.0
module load boost/1_71_0

./kmeans -r 10000000 -c 50 -k 500 -s 1000000 --kpp --lloyd --hybrid --coreset -f "../data/test_10000000_50.txt"