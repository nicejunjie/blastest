 


CUBLAS=/home/nvidia/junjieli//nvhpc/23.11/Linux_aarch64/23.11/math_libs/lib64/libcublas.so
CURT=/home/nvidia/junjieli//nvhpc/23.11/Linux_aarch64/23.11/cuda/lib64/libcudart.so
CUINCLUDE=/home/nvidia/junjieli//nvhpc/23.7/src/nvhpc_2023_237_Linux_aarch64_cuda_multi/install_components/Linux_aarch64/23.7/cuda/12.2/include/
CUINCLUDE=/home/nvidia/junjieli//nvhpc/23.11/Linux_aarch64/23.11/cuda/include

#COPY="-DGPUCOPY -DDEBUG -DCUDA_MEM_POOL"
#COPY="-DGPUCOPY -DCUDA_MEM_POOL"
#COPY="-DDEBUG -DGPUCOPY -DCUDA_ASYNC "
#COPY="-DDEBUG -DGPUCOPY"
COPY="-DAUTO_NUMA "

CC=mpicc 

CFLAGS=" -O2 -lnuma -mp"
EXTRA_FLAGS="--diag_suppress incompatible_assignment_operands --diag_suppress set_but_not_used --diag_suppress incompatible_param"
FLAGS="$CFLAGS $EXTRA_FLAGS"
$CC -c -g -fPIC mysecond.c -o mysecond.o  $FLAGS
$CC $COPY -c -g   -fPIC mylib.c  -o mylib.o  -I$CUINCLUDE -traceback $FLAGS
#$CC -DINIT_IN_MPI $COPY -c -g   -fPIC mylib.c  -o mylib-mpi.o  -I$CUINCLUDE -traceback $FLAGS
$CC -shared -g  -o mylib.so mylib.o mysecond.o $CUBLAS $CURT -traceback $FLAGS 
#$CC -shared -g  -o mylib-mpi.so mylib-mpi.o mysecond.o $CUBLAS $CURT -traceback $FLAGS 

pgfortran -g -mp  -lblas -O2 -Minfo=all test_dgemm.f90 mysecond.o

 NVBLAS=/opt/nvidia/hpc_sdk/Linux_aarch64/23.7/math_libs/12.2/targets/sbsa-linux/lib/libnvblas.so
#export PGI_TERM=trace #debug trace signal abort 


#M=93536
M=20816
N=2400
#M=500
#N=500
K=32
ni=3

echo "-------------------------"
echo Matrix Size: $M $N $K
echo "-------------------------"
export OMP_NUM_THREADS=4
echo ""
echo $M $N $K $ni| ./a.out 
echo ""
echo $M $N $K $ni|LD_PRELOAD=./mylib.so ./a.out   
echo ""
echo $M $N $K $ni|LD_PRELOAD=./mylib.so  numactl -m 1 ./a.out  

#echo $M $N $K $ni|LD_PRELOAD=$NVBLAS ./a.out  
#echo $M $N $K $ni|LD_PRELOAD=$NVBLAS numactl -m 1 ./a.out  
