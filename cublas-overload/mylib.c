 
#define _GNU_SOURCE

//#include <stdlib.h>
//#include <stdio.h>
#include "mylib.h"
#include <cublas_v2.h>
#include <time.h>
#include <math.h>
#include <dlfcn.h>
#include <stdbool.h>


#define GB 1024*1024*1024;
#define MB 1024*1024;
#define KB 1024;


extern double mysecond();
static void (*orig_dgemm)()=NULL; 
cublasStatus_t status;
cublasHandle_t handle;
MemoryPool memoryPool, memoryPool0;
size_t poolSize = (size_t)1*GB; 
bool poolinit=false;

void dgemm_( const char* transa, const char* transb, const int* m, const int* n, const int* k, 
                 const double* alpha, const double* A, const int* lda, const double* B, const int* ldb, 
                 const double* beta, double* C, const int* ldc) {

#ifdef CUDA_MEM_POOL
   if(!poolinit) {
      memoryPool = createMemoryPool(poolSize);
      memoryPool0 = memoryPool;
      poolinit=true;
   }
#endif
#ifdef DEBUG
   double ta1,ta0;
   ta0=mysecond();
   double t1,t0;
   t0=ta0;
#endif

   double avgn=cbrt(*m)*cbrt(*n)*cbrt(*k);
   printf("msize: %d %d %d  mmem: %d MB\n",*m, *n, *k, ((*m)*(*k)+(*k)*(*n)+(*m)*(*n))/1024/1024*8);
   if(avgn<500)  {
         printf("%s\n", "on cpu");
         orig_dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); 
         return;
   }
   printf("%s %.1f\n", "on gpu", avgn);

//   fprintf(stdout,"overloading dgemm_\n");

    // Perform matrix multiplication
   cublasOperation_t transA = (*transa == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
   cublasOperation_t transB = (*transb == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;

#ifdef DEBUG
    t1=mysecond()-t0;
    printf("cudainit   time %.6f\n",t1);
#endif

#ifdef GPUCOPY

#ifdef DEBUG
    t0=mysecond();
#endif
    // Allocate memory on GPU
    double *d_A, *d_B, *d_C;
#ifdef CUDA_MEM_POOL
    d_A = (double*)allocateFromPool(&memoryPool, (*m)*(*k)*sizeof(double));
    d_B = (double*)allocateFromPool(&memoryPool, (*k)*(*n)*sizeof(double));
    d_C = (double*)allocateFromPool(&memoryPool, (*m)*(*n)*sizeof(double));
#else 
    cudaMalloc((void **)&d_A, (*m) * (*k) * sizeof(double));
    cudaMalloc((void **)&d_B, (*k) * (*n) * sizeof(double));
    cudaMalloc((void **)&d_C, (*m) * (*n) * sizeof(double));
#endif

  //  cudaMemset(d_A, 0, (*m) * (*k) * sizeof(double));
  //  cudaMemset(d_B, 0, (*k) * (*n) * sizeof(double));
  //  cudaMemset(d_C, 0, (*m) * (*n) * sizeof(double));

#ifdef DEBUG
    t1=mysecond()-t0;
    printf("cudamalloc time %.6f\n",t1);
#endif
    
#ifdef DEBUG
    t0=mysecond();
#endif
    // Copy from host to GPU
    cudaMemcpy(d_A, A, (*m) * (*k) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, (*k) * (*n) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, (*m) * (*n) * sizeof(double), cudaMemcpyHostToDevice);

#ifdef DEBUG
    t1=mysecond()-t0;
    printf("cudamemcpy time %.6f\n",t1);
#endif

#ifdef DEBUG
    t0=mysecond();
#endif
    status = cublasDgemm(handle, transA, transB, *m, *n, *k, alpha, d_A, *lda, d_B, *ldb, beta, d_C, *ldc);
    cudaDeviceSynchronize();
#ifdef DEBUG
    t1=mysecond()-t0;
    printf("cudablas1  time %.6f\n",t1);
#endif
   
#ifdef DEBUG
    t0=mysecond();
#endif
    cudaMemcpy(C, d_C, (*m) * (*n) * sizeof(double), cudaMemcpyDeviceToHost);
#ifdef DEBUG
    t1=mysecond()-t0;
    printf("cudacpback time %.6f\n",t1);
#endif

    // Free GPU memory
#ifdef DEBUG
    t0=mysecond();
#endif
#ifdef CUDA_MEM_POOL
    memoryPool = memoryPool0;
#else
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
#endif
#ifdef DEBUG
    t1=mysecond()-t0;
    printf("cudafree   time %.6f\n",t1);
#endif

#else  //not GPUCPOY
#ifdef DEBUG
    t0=mysecond();
#endif
    status = cublasDgemm(handle, transA, transB, *m, *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc);
    cudaDeviceSynchronize();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Error in cublasDgemm\n");
    }
#ifdef DEBUG
    t1=mysecond()-t0;
    printf("cudablas2  time %.6f\n",t1);
#endif
#endif


#ifdef DEBUG
    ta1=mysecond()-ta0;
    printf("* my total time %.6f\n",ta1);
#endif
//    cublasDestroy(handle);
    return;
}



void mylib_init(){
    orig_dgemm= dlsym(RTLD_NEXT, "dgemm_");
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        return;
    }
//    dgemm_0call();
    return;
}
void mylib_fini(){
    // Destroy the handle
    cublasDestroy(handle);
#ifdef CUDA_MEM_POOL
    if(poolinit) destroyMemoryPool(&memoryPool);
#endif
    return;
}


  __attribute__((section(".init_array"))) void *__init = mylib_init;
  __attribute__((section(".fini_array"))) void *__fini = mylib_fini;
 
