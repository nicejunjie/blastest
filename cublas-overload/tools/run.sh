 


CC=pgcc

CFLAGS=" -O2 " 
EXTRA_FLAGS=
FLAGS="$CFLAGS $EXTRA_FLAGS"
$CC -c -g -fPIC thpoff-lib.c -o thpoff-lib.o  $FLAGS
$CC -shared -g  -o thpoff.so thpoff-lib.o $FLAGS 
