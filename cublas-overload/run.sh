 

#NVHOME=/scratch/07893/junjieli/soft/nvhpc/23.11/Linux_x86_64/23.11
NV=23.11
NVHOME=/home/nvidia/junjieli/soft/nvhpc/$NV/Linux_aarch64/$NV
CUBLAS=$NVHOME/math_libs/lib64/libcublas.so
CURT=$NVHOME/cuda/lib64/libcudart.so
CUINCLUDE=$NVHOME/cuda/include

#COPY="-DGPUCOPY -DDEBUG -DCUDA_MEM_POOL"
#COPY="-DGPUCOPY -DCUDA_MEM_POOL"
#COPY="-DDEBUG -DGPUCOPY -DCUDA_ASYNC "
#COPY="-DDEBUG -DGPUCOPY"
COPY="-DAUTO_NUMA "

CC=pgcc

CFLAGS=" -O2 -lnuma -mp" # -gpu=unified"
EXTRA_FLAGS="--diag_suppress incompatible_assignment_operands --diag_suppress set_but_not_used --diag_suppress incompatible_param"
FLAGS="$CFLAGS $EXTRA_FLAGS"
$CC -c -g -fPIC mysecond.c -o mysecond.o  $FLAGS
$CC $COPY -c -g   -fPIC mylib.c  -o mylib.o  -I$CUINCLUDE -traceback $FLAGS
$CC -shared -g  -o mylib.so mylib.o mysecond.o $CUBLAS $CURT -traceback $FLAGS 

pgfortran -g -mp  -lblas -O2 -Minfo=all test_dgemm.f90 mysecond.o

 NVBLAS=$NVHOME/math_libs/12.3/targets/sbsa-linux/lib/libnvblas.so
#export PGI_TERM=trace #debug trace signal abort 


M=20816
N=2400
#N=500
K=32
#M=4800 
#N=4800 
#K=748288
ni=3

echo "-------------------------"
echo Matrix Size: $M $N $K
echo "-------------------------"
export OMP_NUM_THREADS=72
echo ""
echo $M $N $K $ni| ./a.out 
echo ""
echo $M $N $K $ni|LD_PRELOAD=./mylib.so ./a.out   
echo ""
echo $M $N $K $ni|LD_PRELOAD=./mylib.so  numactl -m 1 ./a.out  

#echo $M $N $K $ni|LD_PRELOAD=$NVBLAS ./a.out  
#echo $M $N $K $ni|LD_PRELOAD=$NVBLAS numactl -m 1 ./a.out  
