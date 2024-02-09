

#define _GNU_SOURCE 

#include <stdio.h> 
#include <dlfcn.h>
#include <stdlib.h>


void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, 
                const double *alpha, const double *a, const int *lda, const double *b, const int *ldb, 
                const double *beta, double *c, const int *ldc) 
{
   static void (*orig_f)()=NULL;
   if (!orig_f) orig_f = dlsym(RTLD_NEXT, __func__);

   printf("dgemm_ intercepted\n");

   orig_f(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   return;
}

