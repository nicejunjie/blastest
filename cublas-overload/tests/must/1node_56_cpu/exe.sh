#!/bin/bash



export OMP_NUM_THREADS=$1
exe=/home/nvidia/hliu/MuST/MuST_CPU_NVPL/bin/mst2
#exe=/home/nvidia/hliu/MuST/MuST_CPU/bin/mst2
export LD_PRELOAD=/home/nvidia/junjieli/blastest/cublas-overload/zgemm/mylib.so
$exe < i_mst       # Use ibrun instead of mpirun or mpiexec
