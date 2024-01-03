 


CUBLAS=/home/nvidia/junjieli//nvhpc/23.11/Linux_aarch64/23.11/math_libs/lib64/libcublas.so
CURT=/home/nvidia/junjieli//nvhpc/23.11/Linux_aarch64/23.11/cuda/lib64/libcudart.so
CUINCLUDE=/home/nvidia/junjieli//nvhpc/23.7/src/nvhpc_2023_237_Linux_aarch64_cuda_multi/install_components/Linux_aarch64/23.7/cuda/12.2/include/
CUINCLUDE=/home/nvidia/junjieli//nvhpc/23.11/Linux_aarch64/23.11/cuda/include

#COPY="-DGPUCOPY -DDEBUG -DCUDA_MEM_POOL"
#COPY="-DGPUCOPY -DCUDA_MEM_POOL"
 COPY="-DDEBUG "

FLAGS="--diag_suppress incompatible_assignment_operands --diag_suppress set_but_not_used"
pgcc -c -g -fPIC mysecond.c -o mysecond.o  $FLAGS
pgcc $COPY -c -g   -fPIC mylib.c  -o mylib.o  -I$CUINCLUDE -traceback $FLAGS
#pgcc -c -g -fPIC cuda_mem_pool.c -o cuad_mem_pool.o -I$CUINCLUDE -traceback $FLAGS
pgcc -shared -g  -o mylib.so mylib.o mysecond.o $CUBLAS $CURT -traceback $FLAGS 

pgfortran -g -mp  -lblas -O2 -Minfo=all test_dgemm.f90 mysecond.o


#export PGI_TERM=trace #debug trace signal abort 


M=20816
N=2400
K=32
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
