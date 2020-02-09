#!/bin/bash

qsub run_serial1.sh
qsub run_serial2.sh
qsub run_serial3.sh
qsub run_serial4.sh

qsub run_omp1.sh
qsub run_omp2.sh
qsub run_omp3.sh
qsub run_omp4.sh

qsub run_mpi1.sh
qsub run_mpi2.sh
qsub run_mpi3.sh
qsub run_mpi4.sh

qsub run_hybrid1.sh
qsub run_hybrid2.sh
qsub run_hybrid3.sh
qsub run_hybrid4.sh
