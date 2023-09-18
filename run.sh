#!/bin/bash 

rm a.out 

TEST=test_zgemm.f90

  pgcc -c -g mysecond.c -o mysecond.o
  pgfortran -g  -lblas -O2 -Minfo=all $TEST mysecond.o

#pgfortran /opt/nvidia/hpc_sdk/Linux_aarch64/23.5/math_libs/12.1/targets/sbsa-linux/lib/libnvblas.so -O2 -Minfo=all $TEST mysecond.o

  rm *.o

export OMP_NUM_THREADS=72

export NVBLAS_CONFIG_FILE=nvblas.conf
 NVBLAS=/opt/nvidia/hpc_sdk/Linux_aarch64/23.7/math_libs/12.2/targets/sbsa-linux/lib/libnvblas.so
#NVBLAS=/home/nvidia/junjieli/nvhpc/23.5/Linux_aarch64/23.5/math_libs/12.1/targets/sbsa-linux/lib/libnvblas.so




for i in 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 7000 10000 15000 20000 30000 40000
do
  M=$i
  N=$i
  K=$i
  echo $M $N $K 
  echo $M $N $K | ./a.out
  echo $M $N $K| LD_PRELOAD=$NVBLAS ./a.out 2>/dev/null
  if grep -q "gpu" nvblas.log; then
    sleep 1
    echo  "gpu"
  elif grep -q "cpu" nvblas.log; then
    sleep 1
    echo  "cpu"
  fi

  
done

