#!/bin/bash
#$ -cwd
#$ -o ../output/mpi3.txt
#$ -j y
#$ -l h_data=375M,h_rt=10:00:00
#$ -pe dc* 16

. /u/local/Modules/default/init/modules.sh
module load gcc/7.2.0
module load openmpi/3.0.0

./kmeans -r 10000000 -c 50 -k 500 -s 1000000 --kpp --lloyd --coreset --mpi -f "../data/test_10000000_50.txt"