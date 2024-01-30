/*
NVHOME=/scratch/07893/junjieli/soft/nvhpc/23.11/Linux_x86_64/23.11
#NVHOME=/home/nvidia/junjieli//nvhpc/23.11/Linux_aarch64/23.11
CURT=$NVHOME/cuda/lib64/libcudart.so
CUINCLUDE=$NVHOME/cuda/include
pgc++ checkgpu.cpp  -I$CUINCLUDE $CURT
 */


#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

void checkDirectManagedMemoryAccess() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Assuming device 0, change if needed

    if (prop.pageableMemoryAccess) {
        printf("Direct Managed Memory Access is supported.\n");
    } else {
        printf("Direct Managed Memory Access is not supported.\n");
    }
}

int main() {
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);  // Assuming device 0, change if needed
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    // Check for direct managed memory access support
    checkDirectManagedMemoryAccess();

    return 0;
}

