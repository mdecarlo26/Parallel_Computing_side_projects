/* Host code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Build as follows: make clean && make
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

/* Include the kernel code */
#include "jacobi_iteration_kernel.cu"

/* Uncomment the line below if you want the code to spit out debug information. */ 
/* #define DEBUG */

int main(int argc, char **argv) 
{
	if (argc != 3) {
		printf("Usage: ./jacobi_iteration thread_block_size matrix_size\n");
		exit(EXIT_FAILURE);
	}

    int thread_block_size = atoi(argv[1]);
    int mat_size = atoi(argv[2]);

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
	matrix_t gpu_naive_solution_x;  /* Solution computed by naive kernel */
    matrix_t gpu_opt_solution_x;    /* Solution computed by optimized kernel */

	/* Initialize the random number generator */
	srand(time(NULL));

	/* Generate diagonally dominant matrix */ 
    printf("\nGenerating %d x %d system\n", mat_size, mat_size);
	A = create_diagonally_dominant_matrix(mat_size, mat_size);
	if (A.elements == NULL) {
        printf("Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create the other vectors */
    B = allocate_matrix_on_host(mat_size, 1, 1);
	reference_x = allocate_matrix_on_host(mat_size, 1, 0);
	gpu_naive_solution_x = allocate_matrix_on_host(mat_size, 1, 0);
    gpu_opt_solution_x = allocate_matrix_on_host(mat_size, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    /* Compute Jacobi solution on CPU */
	printf("\nPerforming Jacobi iteration on the CPU\n");
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B,mat_size);
    gettimeofday(&stop, NULL);
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	fprintf(stderr, "CPU time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
	+ (stop.tv_usec - start.tv_usec)/(float)1000000));
	/* Compute Jacobi solution on device. Solutions are returned 
       in gpu_naive_solution_x and gpu_opt_solution_x. */
    printf("\nPerforming Jacobi iteration on device\n");
	compute_on_device(A, gpu_naive_solution_x, gpu_opt_solution_x, B,thread_block_size,mat_size);
    display_jacobi_solution(A, gpu_naive_solution_x, B); /* Display statistics */
    display_jacobi_solution(A, gpu_opt_solution_x, B); 
    
    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(gpu_naive_solution_x.elements);
    free(gpu_opt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}


/* FIXME: Complete this function to perform Jacobi calculation on device */
void compute_on_device(const matrix_t A, matrix_t gpu_naive_sol_x, 
                       matrix_t gpu_opt_sol_x, const matrix_t B,int thread_block_size,int mat_size)
{
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    matrix_t A_on_device = allocate_matrix_on_device(A);
    copy_matrix_to_device(A_on_device,A);
    matrix_t B_on_device = allocate_matrix_on_device(B);
    copy_matrix_to_device(B_on_device,B);
    double ssd;
    // *ssd = INFINITY;
    double* ssd_on_device = NULL;
    cudaMalloc((void**)&ssd_on_device,sizeof(double));

    dim3 threads(thread_block_size,1,1);
    dim3 grid( (B.num_rows +thread_block_size - 1)/thread_block_size,1);


    matrix_t x_new = allocate_matrix_on_device(gpu_naive_sol_x);
    copy_matrix_to_device(x_new,B);
    matrix_t x_old = allocate_matrix_on_device(B);
    copy_matrix_to_device(x_old,B);
    matrix_t tmp;
    int done = 0;
    int num_iter = 0;
    double mse;

    gettimeofday(&stop, NULL);
    fprintf(stderr, "GPU Initialization time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
	+ (stop.tv_usec - start.tv_usec)/(float)1000000));
    gettimeofday(&start, NULL);

    printf("Compute Naive Solution\n");
    while(!done){
        /*
        a. cudaMemset() SSD to zero on device.
        b. Launch kernel. For fastest implementation, a single launch can
        update the x vector and calculate the SSD value. Use parallel
        reduction to calculate SSD on device.
        c. Copy SSD back to host from device.
        d. If SSD has converged, done = 1.
        e. Flip pointers for ping-pong buffers on the device.

        */

        cudaMemset(ssd_on_device,0,sizeof(double));//set SSD to zero
        // check_CUDA_error("kernel error");
        jacobi_iteration_kernel_naive<<< grid,threads >>>(A_on_device.elements,x_new.elements,x_old.elements,B_on_device.elements,ssd_on_device,mat_size);
        check_CUDA_error("kernel error");
        cudaDeviceSynchronize();
        cudaMemcpy(&ssd, ssd_on_device,  sizeof(double), cudaMemcpyDeviceToHost);
        // check_CUDA_error("kernel error");
        num_iter++; 
        // printf("ssd: %lf\n",ssd);
        mse = sqrt (ssd);
        // printf("Iteration: %d. MSE = %lf\n", num_iter, mse); 
        if ((mse <= THRESHOLD) || (num_iter == MAX_ITER))
            done = 1;

        tmp.elements = x_old.elements;
        x_old.elements = x_new.elements;
        x_new.elements = tmp.elements;

    }
    printf("\nConvergence achieved after %d iterations. MSE: %lf \n", num_iter,mse);

    copy_matrix_from_device(gpu_naive_sol_x,x_new);

    gettimeofday(&stop, NULL);
    fprintf(stderr, "GPU Naive time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
	+ (stop.tv_usec - start.tv_usec)/(float)1000000));
    gettimeofday(&start, NULL);

    
    done = 0;
    mse = 0;
    num_iter = 0;
    matrix_t A_trans = makeTranspose(A);
    copy_matrix_to_device(x_new,B);
    copy_matrix_to_device(x_old,B);
    copy_matrix_to_device(A_on_device,A_trans);
    printf("Compute Optimal Solution\n");
    gettimeofday(&start, NULL);
    while(!done){
        /*
        a. cudaMemset() SSD to zero on device.
        b. Launch kernel. For fastest implementation, a single launch can
        update the x vector and calculate the SSD value. Use parallel
        reduction to calculate SSD on device.
        c. Copy SSD back to host from device.
        d. If SSD has converged, done = 1.
        e. Flip pointers for ping-pong buffers on the device.

        */

        cudaMemset(ssd_on_device,0,sizeof(double));//set SSD to zero
        // check_CUDA_error("kernel error");
        jacobi_iteration_kernel_optimized<<< grid,threads >>>(A_on_device.elements,x_new.elements,x_old.elements,B_on_device.elements,ssd_on_device,mat_size);
        check_CUDA_error("kernel error");
        cudaDeviceSynchronize();
        cudaMemcpy(&ssd, ssd_on_device,  sizeof(double), cudaMemcpyDeviceToHost);
        // check_CUDA_error("kernel error");
        num_iter++; 
        // printf("ssd: %lf\n",ssd);
        mse = sqrt (ssd);
        // printf("Iteration: %d. MSE = %lf\n", num_iter, mse); 
        if ((mse <= THRESHOLD) || (num_iter == MAX_ITER))
            done = 1;

        tmp.elements = x_old.elements;
        x_old.elements = x_new.elements;
        x_new.elements = tmp.elements;

    }


    printf("\nConvergence achieved after %d iterations. MSE: %lf \n", num_iter,mse);

    copy_matrix_from_device(gpu_opt_sol_x,x_new);

    gettimeofday(&stop, NULL);
    fprintf(stderr, "GPU Optimal time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
	+ (stop.tv_usec - start.tv_usec)/(float)1000000));
    gettimeofday(&start, NULL);

    free_matrix_on_device(x_new);
    free_matrix_on_device(x_old);
    free_ssd_on_device(ssd_on_device);
    free_matrix_on_device(A_on_device);
    free_matrix_on_device(B_on_device);
    check_CUDA_error("kernel error");
    free(A_trans.elements);

    gettimeofday(&stop, NULL);
    fprintf(stderr, "GPU free time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
	+ (stop.tv_usec - start.tv_usec)/(float)1000000));
    return;
}


matrix_t makeTranspose(matrix_t A){
    matrix_t A_trans = allocate_matrix_on_host(A.num_rows,A.num_columns,0);
    int i, j;
    for (i = 0; i < A_trans.num_rows; i++) {
       for (j = 0; j < A_trans.num_columns; j++) {
           A_trans.elements[i * A_trans.num_columns + j] = A.elements[j * A_trans.num_columns + i];
       } 
    }
    return A_trans;
}



void free_ssd_on_device(double* ssd){
    cudaFree(ssd);
}




/*Free memory on device*/
void free_matrix_on_device(matrix_t mat){
    cudaFree(mat.elements);
}

/* Allocate matrix on the device of same size as M */
matrix_t allocate_matrix_on_device(const matrix_t M)
{
    matrix_t Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void **)&Mdevice.elements, size);
    return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix_on_host(int num_rows, int num_columns, int init)
{	
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float *)malloc(size * sizeof(float));
	for (unsigned int i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

/* Copy matrix to device */
void copy_matrix_to_device(matrix_t Mdevice, const matrix_t Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device to host */
void copy_matrix_from_device(matrix_t Mhost, const matrix_t Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    return;
}

// void copy_ssd_from_device(double* ssd_host, double* ssd_device ){
//     cudaMemcpy(ssd_host,ssd_device,sizeof(double),cudaMemcpyDeviceToHost);
//     return;
// }

/* Prints the matrix out to screen */
void print_matrix(const matrix_t M)
{
	for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++) {
			printf("%f ", M.elements[i * M.num_columns + j]);
        }
		
        printf("\n");
	} 
	
    printf("\n");
    return;
}

/* Returns a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check for errors in kernel execution */
void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
    
    return;    
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(unsigned int num_rows, unsigned int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));
    if (M.elements == NULL)
        return M;

	/* Create a matrix with random numbers between [-.5 and .5] */
    unsigned int i, j;
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
	for (i = 0; i < num_rows; i++) {
		float row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    return M;
}

