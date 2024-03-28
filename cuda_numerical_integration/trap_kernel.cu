/* GPU kernel to estimate integral of the provided function using the trapezoidal rule. */

/* Device function which implements the function. Device functions can be called from within other __device__ functions or __global__ functions (the kernel), but cannot be called from the host. */ 


#define THREAD_BLOCK_SIZE 512          /* Size of a thread block */
#define NUM_BLOCKS 40                   /* Number of thread blocks */


__device__ float fd(float x) 
{
     return sqrtf((1 + x*x)/(1 + x*x*x*x));
}

/* Kernel function */
__global__ void trap_kernel(float a, float b, long int n, float h,double* result) 
{
     __shared__ double sum_per_thread[THREAD_BLOCK_SIZE];    /* Shared memory for thread block */
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  /* Obtain thread index */
	int stride = blockDim.x * gridDim.x;
     double sum = 0.0f; 
	unsigned int i = thread_id; 
     if(i==0){
          i+=stride;
          sum +=(fd(a)+fd(b))/(2.0);
     } 


     while (i < n) {
		sum += fd(a+i*h);
		i += stride;
	}

     sum_per_thread[threadIdx.x] = sum*h;
     __syncthreads();
     i = blockDim.x/2;

     while (i != 0) {
		if (threadIdx.x < i) 
			sum_per_thread[threadIdx.x] += sum_per_thread[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}

	/* Accumulate sum computed by this thread block into the global shared variable. 
     * This version uses atomicAdd() operation. */
	if (threadIdx.x == 0) {
          // sum_per_thread[0] += (fd(a)+fd(b))/(2.0) * h;
        atomicAdd(result, sum_per_thread[0]);
	}
     return;
}
