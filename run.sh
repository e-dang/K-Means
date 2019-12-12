#!/bin/bash
#$ -cwd
#$ -o ./out.txt
#$ -j y
#$ -l h_data=1G,h_rt=00:01:00
#$ -pe dc* 16

. /u/local/Modules/default/init/modules.sh
module load intel/13.cs
module load gcc/7.2.0
module load openmpi/3.0.0

$MPI_BIN/mpiexec --prefix $MPI_DIR -n 16 ./kmeans