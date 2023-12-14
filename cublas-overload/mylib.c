 
#define _GNU_SOURCE

#include <stdio.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <dlfcn.h>

extern double mysecond();
static void (*orig_dgemm)()=NULL; 
cublasStatus_t status;
cublasHandle_t handle;

void dgemm_( const char* transa, const char* transb, const int* m, const int* n, const int* k, 
                 const double* alpha, const double* A, const int* lda, const double* B, const int* ldb, 
                 const double* beta, double* C, const int* ldc) {

#ifdef DEBUG
   double ta1,ta0;
   ta0=mysecond();
   double t1,t0;
#endif
   int avgn=cbrt(*m)*cbrt(*n)*cbrt(*k);
   if(avgn<10)  {
        printf("%s\n", "on cpu");
         orig_dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); 
         return;
   }
   printf("%s %.1f\n", "on gpu", avgn);

    fprintf(stdout,"overloading dgemm_\n");

    // Perform matrix multiplication
    cublasOperation_t transA = (*transa == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transB = (*transb == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;

#ifdef GPUCOPY

#ifdef DEBUG
    t0=mysecond();
#endif
    // Allocate memory on GPU
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, (*m) * (*k) * sizeof(double));
    cudaMalloc((void **)&d_B, (*k) * (*n) * sizeof(double));
    cudaMalloc((void **)&d_C, (*m) * (*n) * sizeof(double));
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
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
#ifdef DEBUG
    t1=mysecond()-t0;
    printf("cudafree   time %.6f\n",t1);
#endif
#else 
#ifdef DEBUG
    t0=mysecond();
#endif
    status = cublasDgemm(handle, transA, transB, *m, *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc);
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
    cublasDestroy(handle);
    return;
}


void mylib_init(){
    orig_dgemm= dlsym(RTLD_NEXT, "dgemm_");
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        return;
    }

}
void mylib_fini(){
    // Destroy the handle
  //  cublasDestroy(handle);
    return;
}


  __attribute__((section(".init_array"))) void *__init = mylib_init;
  __attribute__((section(".fini_array"))) void *__fini = mylib_fini;

