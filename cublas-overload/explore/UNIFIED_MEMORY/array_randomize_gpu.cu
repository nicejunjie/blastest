#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void randomArray(float* gpu_array, unsigned long SIZE, unsigned seed) {
        //random number generator seed
	curandState gpu_curand_state;
	curand_init(seed, blockIdx.x, threadIdx.x, &gpu_curand_state);
        //global_index: the global index in the gpu_array
        unsigned global_index;
        //niter: number of iterations for each GPU thread
        unsigned long niter = SIZE/(gridDim.x*blockDim.x);
        for(unsigned long i = 0;i<niter;i++){
                global_index = (blockIdx.x*blockDim.x+threadIdx.x)*niter+i;
                //gpu_array[global_index] = global_index;
                gpu_array[global_index] = curand_uniform(&gpu_curand_state); 
	}
}

int main(int argc, char *argv[]) {

	/* set the ramdon seed as the curret time */
        unsigned seed = time(0);
	/* set the array size to the first command-line argument */ 
	unsigned long SIZE = (unsigned long) (atoi(argv[1]));
	printf("\nRamdomnizing an array with %ld elements\n", SIZE);
	/* set the grid size to the second command-line argument */ 
	unsigned long gridsize = (unsigned long) (atoi(argv[2]));
	/* set the block size to the third command-line argument */ 
	unsigned long blocksize = (unsigned long) (atoi(argv[3]));
	if ( SIZE%(gridsize*blocksize) !=0 ) {
		printf("Error! Array size of %ld is NOT divisible by %ld * %ld \n",SIZE, gridsize, blocksize); 
		printf("Exiting with an error code of -1\n\n");
		exit(-1);
	}

        printf("\n\nConventional CUDA Memory Access\n");
        /* starting time */
        clock_t t0 = clock();

	/* allocate the GPU array */
	float* cpu_array = (float*) malloc(SIZE * sizeof(float));
	/*randomArray(cpu_array, SIZE);*/
	/* allocate the GPU array */
	float* gpu_array;
	cudaMalloc((void**) &gpu_array, SIZE * sizeof(float));

        /* starting time of GPU kernel execution */
        clock_t t1 = clock();

        /* randomize the GPU array */
        randomArray<<<gridsize,blocksize>>>(gpu_array, SIZE, seed);	
        /* synchronize GPU cores */
        cudaDeviceSynchronize();

	/* ending time of GPU kernel execution*/
        clock_t t2 = clock();
        /* elapsed time of GPU kernel execution */
        double t2_t1=(t2-t1)/(double) CLOCKS_PER_SEC;
        printf("\nElasped time of GPU kernel execution: %f seonds\n", t2_t1);

        /* starting time of GPU-to-CPU data transfer*/
        clock_t t3 = clock();
	/* copy the gpu array to its cpu counterpart */
	//cudaMemcpy(cpu_array, gpu_array,SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_array, gpu_array,SIZE*sizeof(float), cudaMemcpyDefault);
        /* ending time of GPU-to-CPU data transfer*/
        clock_t t4 = clock();
        /* elapsed time of GPU-to-CPU data execution */
        double t4_t3=(t4-t3)/(double) CLOCKS_PER_SEC;
        printf("\nElasped Time of GPU-to-CPU data transfer: %f seonds\n", t4_t3);
        /*for(unsigned long i = 0; i<SIZE; i++){
               printf("CPU ARRAY %ld : %f\n",i,cpu_array[i]); 
        }*/

	/* deallocate the gpu array */
        cudaFree(gpu_array);			
	/* deallocate the cpu array */
	free(cpu_array);

        printf("\n\nZero-Copy CUDA Memory Access\n");
        /* allocate the host array */
        float* host_array = NULL;
        cudaHostAlloc((void**) &host_array, SIZE*sizeof(float), cudaHostAllocMapped);

        /* assigne the device array to the host array */
        float* device_array;
        cudaHostGetDevicePointer((void **) &device_array, (void *) host_array,0);

        /* starting time of GPU kernel execution*/
        clock_t t5 = clock();
      
        /* randomize the device array */
        randomArray<<<gridsize,blocksize>>>(device_array, SIZE, seed);
        /* synchronize GPU cores */
        cudaDeviceSynchronize();

        /*for(unsigned long i = 0; i<SIZE; i++){
            printf("Host ARRAY %ld : %f\n",i,host_array[i]); 
        }*/
        /* ending time of GPU kernel exeucution*/
        clock_t t6 = clock();
        /* elapsed time of GPU kernel execution */
        double t6_t5=(t6-t5)/(double) CLOCKS_PER_SEC;
        printf("\nElasped Time of GPU kernel execution: %f seonds\n", t6_t5);        

	/* deallocate the device array */
        cudaFree(device_array);
	/* deallocate the host array */
        cudaFreeHost(host_array); 

        printf("\n\nUnified CPU/GPU Memory Access\n");
        /* allocate the unified array that can be accessed by either CPU or GPU */
        float* unified_array = NULL;
        cudaMallocManaged(&unified_array, SIZE*sizeof(float));

        /* starting time of GPU kernel execution*/
        clock_t t7 = clock();

        /* randomize the device array */
        randomArray<<<gridsize,blocksize>>>(unified_array, SIZE, seed);
        /* synchronize GPU cores */
        cudaDeviceSynchronize();

        /*for(unsigned long i = 0; i<SIZE; i++){
            printf("Unified ARRAY %ld : %f\n",i,unified_array[i]); 
        }*/
        /* ending time of GPU kernel exeucution*/
        clock_t t8 = clock();
        /* elapsed time of GPU kernel execution */
        double t8_t7=(t8-t7)/(double) CLOCKS_PER_SEC;
        printf("\nElasped Time of GPU kernel execution: %f seonds\n", t8_t7);

        cudaFree(unified_array);
        printf("\n\nMission Accomplished!\n\n");
	return 0;
}
