#!/bin/bash



#export LD_LIBRARY_PATH=/opt/nvidia/nvpl/23.09-alpha/lib:$LD_LIBRARY_PATH 

export LD_PRELOAD=/home/nvidia/junjieli/blastest/cublas-overload/zgemm/mylib.so
#exe=/home/nvidia/hliu/MuST/MuST_CPU/bin/mst2
exe=/home/nvidia/hliu/MuST/MuST_CPU_NVPL/bin/mst2

rank=28
nt=1
mpirun --mca coll ^hcoll -np $rank -map-by node:PE=$nt $exe < i_mst       # Use ibrun instead of mpirun or mpiexec
