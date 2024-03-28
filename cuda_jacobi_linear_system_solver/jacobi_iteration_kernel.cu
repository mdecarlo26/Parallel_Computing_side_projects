#include "jacobi_iteration.h"
// #ifndef _JACOBI_ITERATION_KERNEL_H_
// #define _JACOBI_ITERATION_KERNEL_H_

__global__ void jacobi_iteration_kernel_naive(float* A,float *x_new,float* x_old,float* B,double* ssd,int arr_size)
{
    __shared__ double ssd_per_thread[THREAD_BLOCK_SIZE];    /* Shared memory for thread block */
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  /* Obtain thread index */
	// int stride = blockDim.x * gridDim.x;                    /* Stride for each thread. */
    unsigned int i, j;
    double sum;
    i = thread_id;
    if(i >= arr_size ){
        ssd_per_thread[i] = 0.0;
        return;
    }
    
    sum = -A[i * arr_size + i] * x_old[i];
    for (j = 0; j < arr_size; j++)
        sum += A[i * arr_size + j] * x_old[j];


    x_new[i] = (B[i] - sum)/A[i * arr_size + i];

    ssd_per_thread[threadIdx.x] = (x_new[i] - x_old[i])*(x_new[i] - x_old[i]);
	
    __syncthreads();                       /* Wait for all threads in thread block to finish */

    i = blockDim.x / 2;
	while (i > 0) {
		if (threadIdx.x < i) 
			ssd_per_thread[threadIdx.x] += ssd_per_thread[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}

	/* Accumulate sum computed by this thread block into the global shared variable.
     * This kernel uses atomicCAS() to lock down the shared variable */
	if (threadIdx.x == 0) {
        atomicAdd(ssd, ssd_per_thread[0]);
	}


    return;
}

__global__ void jacobi_iteration_kernel_optimized(float* A,float *x_new,float* x_old,float* B,double* ssd,int arr_size)
{
    __shared__ double ssd_per_thread[THREAD_BLOCK_SIZE];    /* Shared memory for thread block */
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  /* Obtain thread index */
	// int stride = blockDim.x * gridDim.x;                    /* Stride for each thread. */
    unsigned int i, j;
    double sum;
    i = thread_id;
    if(i >= arr_size ){
        ssd_per_thread[i] = 0.0;
        return;
    }
    
    sum = -A[i * arr_size + i] * x_old[i];
    for (j = 0; j < arr_size; j++)
        sum += A[i + arr_size * j] * x_old[j];


    x_new[i] = (B[i] - sum)/A[i * arr_size + i];

    ssd_per_thread[threadIdx.x] = (x_new[i] - x_old[i])*(x_new[i] - x_old[i]);
	
    __syncthreads();                       /* Wait for all threads in thread block to finish */

    i = blockDim.x / 2;
	while (i > 0) {
		if (threadIdx.x < i) 
			ssd_per_thread[threadIdx.x] += ssd_per_thread[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}

	/* Accumulate sum computed by this thread block into the global shared variable.
     * This kernel uses atomicCAS() to lock down the shared variable */
	if (threadIdx.x == 0) {
        atomicAdd(ssd, ssd_per_thread[0]);
	}


    return;
}

// #endif