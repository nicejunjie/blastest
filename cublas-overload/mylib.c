 
#define _GNU_SOURCE

//#include <stdlib.h>
//#include <stdio.h>
#include "mylib.h"
#include <cublas_v2.h>
#include <time.h>
#include <math.h>
#include <dlfcn.h>
#include <stdbool.h>

// disable THP
#include <sys/prctl.h> 

#ifdef INIT_IN_MPI
#include <mpi.h>
#endif


#define GiB 1024*1024*1024;
#define MiB 1024*1024;
#define KiB 1024;


extern double mysecond();

#include <numaif.h>
#include <numa.h>
#include <sys/mman.h>
#include <unistd.h>
#define NUMA_HBM 1
//#define PAGE_SIZE sysconf(_SC_PAGESIZE)

int which_numa(double *var) {
 void * ptr_to_check = var;
 int status[1];
 int ret_code;
 status[0]=-1;
 ret_code=move_pages(0 /*self memory */, 1, &ptr_to_check, NULL, status, 0);
 // this print may cause extra NUMA traffic
 // if(debug) printf("Memory at %p is at numa node %d (retcode %d)\n", ptr_to_check, status[0], ret_code);
 return status[0];
}
void move_numa2(double *ptr, unsigned long size, int target_node) {
  double tnuma=mysecond();
  int status[1];
  status[0]=-1;
  int PAGE_SIZE=getpagesize();
  unsigned long num_pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
#pragma omp parallel for
  for (unsigned long i = 0; i < num_pages; i++) {
     void *page_addr = ptr + (i * PAGE_SIZE / sizeof(double));
     move_pages(0 /*self memory */, 1, &page_addr, &target_node, status, 0);
  }
  tnuma=mysecond()-tnuma;
  printf("move_numa time %15.6f of %lu pages\n", tnuma, num_pages);
  return ;
}
void move_numa(double *ptr, unsigned long size, int target_node) {
// size in Bytes
    //printf("size in move_numa=%d, array size=%d\n",size, size/8);
    double tnuma=mysecond();
    int PAGE_SIZE = getpagesize();
    unsigned long num_pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
    int *status = malloc(num_pages*sizeof(int));
    int *nodes = malloc(num_pages*sizeof(int));
    // Allocate an array to store page addresses
    void **page_addrs = malloc(num_pages * sizeof(void *));
    if (page_addrs == NULL) {
        // Handle allocation failure
        return;
    }

    // Populate the array with page addresses
    for (unsigned long i = 0; i < num_pages; i++) {
        page_addrs[i] = ptr + (i * PAGE_SIZE / sizeof(double));
        nodes[i]=target_node;
        status[i]=-1;
    }

    // Call move_pages once with the array of page addresses
    move_pages(0 /*self memory*/, num_pages, page_addrs, nodes, status, 0);
    //printf("status code\n");
    //for (int i =0; i<num_pages; i++) printf("%d:%d\n",i,status[i]);

    // Free the allocated array
    free(page_addrs);

    tnuma=mysecond()-tnuma;
    //printf("element numa\n");
    //for (int i =0; i<size/8; i++) printf("%d %d\n",i,which_numa(ptr+i));
    printf("move_numa time %15.6f of %lu pages\n", tnuma, num_pages);
    return;
}


static void (*orig_dgemm)()=NULL; 
//static void (*orig_pdgemm)()=NULL; 
//static void (*orig_dgemv)()=NULL; 
cublasStatus_t status;
cublasHandle_t handle;

#ifdef CUDA_ASYNC
cudaStream_t stream;
#endif

#ifdef CUDA_MEM_POOL
MemoryPool memoryPool, memoryPool0;
size_t poolSize = (size_t)1*GB; 
bool poolinit=false;
#endif

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
   unsigned long dgemm_mem_size_bytes = ((unsigned long)(*m) * (*k) + (unsigned long)(*k) * (*n) + (unsigned long)(*m) * (*n)) * sizeof(double);
   unsigned long dgemm_mem_size_mb = dgemm_mem_size_bytes / (1024 * 1024);

   printf("dgemm msize: %d %d %d, mmem: %lu MiB\n",*m, *n, *k, dgemm_mem_size_mb);
   if(avgn<500)  {
         printf("%s %.1f\n", "dgemm on cpu", avgn);
         orig_dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); 
         return;
   }
   printf("%s %.1f\n", "dgemm on gpu", avgn);

//   fprintf(stdout,"overloading dgemm_\n");

    // Perform matrix multiplication
    //cublasOperation_t transA = (*transa == 'N' || *transa == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    //cublasOperation_t transB = (*transb == 'N' || *transb == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transA = (transa[0] == 'N' || transa[0] == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transB = (transb[0] == 'N' || transb[0] == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;


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
#elif defined(CUDA_ASYNC)
    cudaMallocAsync((void **)&d_A, (*m) * (*k) * sizeof(double),stream);
    cudaMallocAsync((void **)&d_B, (*k) * (*n) * sizeof(double),stream);
    cudaMallocAsync((void **)&d_C, (*m) * (*n) * sizeof(double),stream);
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
#elif defined(CUDA_ASYNC)
    cudaFreeAsync(d_A,stream);
    cudaFreeAsync(d_B,stream);
    cudaFreeAsync(d_C,stream);
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

#ifdef AUTO_NUMA
    int inumaA=which_numa(A);
    int inumaB=which_numa(B);
    int inumaC=which_numa(C);
    //printf("numa node of A=%d B=%d C=%d\n", inumaA, inumaB, inumaC);    
    if ( inumaA == 0 ) move_numa(A,(unsigned long)(*m)*(*k)*sizeof(double),NUMA_HBM);
    if ( inumaB == 0 ) move_numa(B,(unsigned long)(*k)*(*n)*sizeof(double),NUMA_HBM);
    if ( inumaC == 0 ) move_numa(C,(unsigned long)(*m)*(*n)*sizeof(double),NUMA_HBM);
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

// disable THP for auto-page migration
#ifdef AUTO_NUMA
    prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
#endif

// register functions
    orig_dgemm= dlsym(RTLD_NEXT, "dgemm_");
    //orig_pdgemm= dlsym(RTLD_NEXT, "pdgemm_");
    //orig_dgemv= dlsym(RTLD_NEXT, "dgemv_");

    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        return;
    }

#ifdef CUDA_ASYNC
       // Create CUDA stream
    cudaStreamCreate(&stream);
#endif
    return;
}
void mylib_fini(){
    cublasDestroy(handle);
#ifdef CUDA_ASYNC
    cudaStreamDestroy(stream);
#endif
#ifdef CUDA_MEM_POOL
    if(poolinit) destroyMemoryPool(&memoryPool);
#endif
    return;
}


  __attribute__((section(".init_array"))) void *__init = mylib_init;
  __attribute__((section(".fini_array"))) void *__fini = mylib_fini;
 
