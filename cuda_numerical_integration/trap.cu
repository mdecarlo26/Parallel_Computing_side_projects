/* Reference code implementing numerical integration.
 *
 * Build and execute as follows: 
        make clean && make 
        ./trap a b n 

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include "cuda_runtime.h"

/* Include the kernel code */
#include "trap_kernel.cu"

double compute_on_device(float, float, long int, float);
extern "C" double compute_gold(float, float, int, float);

//helper funcs for GPU work
void check_for_error(const char *);

int main(int argc, char **argv) 
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s a b n\n", argv[0]);
        fprintf(stderr, "a: Start limit. \n");
        fprintf(stderr, "b: end limit\n");
        fprintf(stderr, "n: number of trapezoids\n");
        exit(EXIT_FAILURE);
    }

    int a = atoi(argv[1]); /* Left limit */
    int b = atoi(argv[2]); /* Right limit */
    long int n = atoi(argv[3]); /* Number of trapezoids */ 

    float h = (b-a)/(float)n; // Height of each trapezoid  
	printf("Number of trapezoids = %d\n", n);
    printf("Height of each trapezoid = %f \n", h);
    struct timeval start, stop;	

    gettimeofday(&start, NULL);

	double reference = compute_gold(a, b, n, h);
    printf("Reference solution computed on the CPU = %f \n", reference);
    gettimeofday(&stop, NULL);
    printf("CPU time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Write this function to complete the trapezoidal on the GPU. */
	double gpu_result = compute_on_device(a, b, n, h);
	printf("Solution computed on the GPU = %f \n", gpu_result);
} 

/* Complete this function to perform the trapezoidal rule on the GPU. */
double compute_on_device(float a, float b, long int n, float h)
{
    double *result_on_device = NULL;
    struct timeval start, stop;	

    gettimeofday(&start, NULL);

    cudaMalloc((void**)&result_on_device, sizeof(double));
	cudaMemset(result_on_device, 0.0f, sizeof(double));

    gettimeofday(&stop, NULL);
    printf("Data transfer time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

    dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1); 
	dim3 grid(NUM_BLOCKS,1);

    gettimeofday(&start, NULL);

    trap_kernel<<<grid, thread_block>>>(a,b,n,h,result_on_device);
    cudaDeviceSynchronize();

    gettimeofday(&stop, NULL);
    printf("Kernel execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

    check_for_error("KERNEL FAILURE");

    double integral = 0.0;
	cudaMemcpy(&integral, result_on_device, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(result_on_device);
    
    return integral;
}

void check_for_error (const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 


