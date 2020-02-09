#!/bin/bash
#$ -cwd
#$ -o ../output/serial3.txt
#$ -j y
#$ -l h_data=6000M,h_rt=10:00:00

. /u/local/Modules/default/init/modules.sh
module load gcc/7.2.0

./kmeans -r 10000000 -c 50 -k 500 -s 1000000 --kpp --lloyd --coreset --serial -f "../data/test_10000000_50.txt"