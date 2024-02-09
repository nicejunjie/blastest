#!/bin/bash 

# simplest dlsym based overload for dgemm
gcc  -g -fPIC -shared   mydgemm.c -o my.so 

EXE=test_dgemm
#dgemm test code
#ifort -qopenmp -O2 -g -qmkl=sequential -o $EXE test_dgemm.f90
pgcc -mp -O2 -g -lblas -o $EXE test_dgemm.f90


M=10
N=10
K=10
Niter=10

echo $M $N $K $Niter | LD_PRELOAD=./my.so  ./$EXE
