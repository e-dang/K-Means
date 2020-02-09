#!/bin/bash
#$ -cwd
#$ -o ../output/mpi2.txt
#$ -j y
#$ -l h_data=375M,h_rt=10:00:00
#$ -pe dc* 16

. /u/local/Modules/default/init/modules.sh
module load gcc/7.2.0
module load openmpi/3.0.0

./kmeans -r 10000000 -c 50 -k 500 --optkpp --optlloyd --mpi -f "../data/test_10000000_50.txt"