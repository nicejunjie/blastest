// PUBLIC DOMAIN CODE
//
// A tiny program that disable transparent huge pages on arbitrary processes
//pgcc -c -g -fPIC thpoff-lib.c -o thpoff-lib.o -O2
//pgcc -shared -g  -o thpoff.so thpoff-lib.o -O2
#include <stdio.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <errno.h>

void mylib_init(){
    printf("THP OFF\n");
    prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
    return;
}

void mylib_fini(){
    return;
}

  __attribute__((section(".init_array"))) void *__init = mylib_init;
  __attribute__((section(".fini_array"))) void *__fini = mylib_fini;
 
