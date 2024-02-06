#!/bin/bash

source ~/junjieli/setup.sh

rank=28
nt=1
# mpirun --mca coll ^hcoll -np $rank  -map-by node:PE=$nt numactl -m 0 ./exe.sh $nt       # Use ibrun instead of mpirun or mpiexec
 mpirun --mca coll ^hcoll -np $rank numactl -m 0 ./exe.sh $nt       # Use ibrun instead of mpirun or mpiexec

mv o_n0000000_CoCrFeMnNi o_n0000000_CoCrFeMnNi-gpu${rank}x${nt}
