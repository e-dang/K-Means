#!/bin/bash
#$ -cwd
#$ -o ../output/serial1.txt
#$ -j y
#$ -l h_data=6000M,h_rt=10:00:00

. /u/local/Modules/default/init/modules.sh
module load gcc/7.2.0

./kmeans -r 10000000 -c 50 -k 500 --kpp --lloyd --serial -f "../data/test_10000000_50.txt"

