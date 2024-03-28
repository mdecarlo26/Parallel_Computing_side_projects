/* Code for the Jacbi equation solver. 
 * Author: Naga Kandasamy
 * Date modified: February 2, 2022
 *
 * Compile as follows:
 * gcc -o solver solver.c solver_gold.c -O3 -Wall -std=c99 -lm -lpthread
 *
 * If you wish to see debug info add the -D DEBUG option when compiling the code.
 */

// System includes
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <pthread.h>
#include "grid.h"
#include <sys/time.h>

extern int compute_gold_gauss(grid_t *);
extern int compute_gold_jacobi(grid_t *);
int compute_using_pthreads_jacobi(grid_t *, int);
void compute_grid_differences(grid_t *, grid_t *);
grid_t *create_grid(int, float, float);
grid_t *copy_grid(grid_t *);
void print_grid(grid_t *);
void print_stats(grid_t *);
double grid_mse(grid_t *, grid_t *);
void *jacobi_worker(void*);


typedef struct thread_payload {
    int tid;
    int num_threads;   
    int chunk_size;
    float eps;
    int p_num_elements; 
    int num_iter;
    int offset;
	grid_t *src_grid;
    grid_t *dest_grid;
} thread_payload;

pthread_barrier_t barrier;  
pthread_mutex_t mutex;

int converged = 0;
double diff = 0; 
int num_elements = 0;

int main(int argc, char **argv)
{	
	if (argc < 5) {
        fprintf(stderr, "Usage: %s grid-dimension num-threads min-temp max-temp\n", argv[0]);
        fprintf(stderr, "grid-dimension: The dimension of the grid\n");
        fprintf(stderr, "num-threads: Number of threads\n"); 
        fprintf(stderr, "min-temp, max-temp: Heat applied to the north side of the plate is uniformly distributed between min-temp and max-temp\n");
        exit(EXIT_FAILURE);
    }
    /* Parse command-line arguments. */
    int dim = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    float min_temp = atof(argv[3]);
    float max_temp = atof(argv[4]);
    
    /* Generate grids and populate them with initial conditions. */
 	grid_t *grid_1 = create_grid(dim, min_temp, max_temp);
    /* Grids 2 and 3 should have the same initial conditions as Grid 1. */
    grid_t *grid_2 = copy_grid(grid_1);
    grid_t *grid_3 = copy_grid(grid_1);

    struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
    
	/* Compute reference solutions using the single-threaded versions. */
    int num_iter;

    gettimeofday(&start, NULL);
	fprintf(stderr, "\nUsing the single threaded version of Gauss to solve the grid\n");
	num_iter = compute_gold_gauss(grid_1);
	fprintf(stderr, "Convergence achieved after %d iterations\n", num_iter);
    /* Print key statistics for the converged values. */
	fprintf(stderr, "Printing statistics for the interior grid points\n");
    print_stats(grid_1);
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));
	
    gettimeofday(&start, NULL);
	fprintf(stderr, "\nUsing the single threaded version of Jacobi to solve the grid\n");
	num_iter = compute_gold_jacobi(grid_2);
	fprintf(stderr, "Convergence achieved after %d iterations\n", num_iter);
    /* Print key statistics for the converged values. */
	fprintf(stderr, "Printing statistics for the interior grid points\n");
    print_stats(grid_2);
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    gettimeofday(&start, NULL);
	/* Use pthreads to solve the equation using the jacobi method. */
	fprintf(stderr, "\nUsing pthreads to solve the grid using the jacobi method\n");
	num_iter = compute_using_pthreads_jacobi(grid_3, num_threads);
	fprintf(stderr, "Convergence achieved after %d iterations\n", num_iter);			
    fprintf(stderr, "Printing statistics for the interior grid points\n");
	print_stats (grid_3);
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));
    
    /* Compute grid differences. */
    fprintf(stderr, "MSE between the single-threaded Gauss and Jacobi grids: %f\n", grid_mse(grid_1, grid_2));
    fprintf(stderr, "MSE between the single-threaded Jacobi and multi-threaded Jacobi grids: %f\n", grid_mse(grid_2, grid_3));

	/* Free up the grid data structures. */
	free((void *) grid_1->element);	
	free((void *) grid_1); 
	free((void *) grid_2->element);	
	free((void *) grid_2);
    free((void *) grid_3);
    free((void *) grid_3->element);

	exit(EXIT_SUCCESS);
}

/* FIXME: Edit this function to use the jacobi method of solving the equation. The final result should be placed in the grid data structure. */
int compute_using_pthreads_jacobi(grid_t *grid, int num_threads)
{		
    grid_t *src_grid = grid;
    grid_t *dest_grid = copy_grid(grid);
    grid_t *to_delete= dest_grid;
    int num_iter = 0;
    float eps = 1e-6;
    pthread_t* thread_id = malloc(num_threads*sizeof(pthread_t));
    thread_payload *worker_thread = malloc(num_threads * sizeof(thread_payload));
    if ((worker_thread == NULL || (thread_id == NULL))){
        fprintf(stderr,"malloc failed when making thread args\n");
        exit(EXIT_FAILURE);
    }
    pthread_mutex_init(&mutex, NULL);
    pthread_barrier_init(&barrier, NULL, num_threads);
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    int chunk_size = floor((grid->dim-2)/num_threads);

    int j;
    for (j = 0; j < num_threads; j++){
        worker_thread[j].chunk_size = chunk_size;
        worker_thread[j].offset = j * chunk_size+1;
        worker_thread[j].num_threads = num_threads;
        worker_thread[j].src_grid = src_grid;
        worker_thread[j].dest_grid = dest_grid;
        worker_thread[j].eps = eps;
        worker_thread[j].num_iter = 0;
        worker_thread[j].tid = j;
        worker_thread[j].p_num_elements = 0;
        pthread_create(&thread_id[j],&attributes,jacobi_worker,(void*) &worker_thread[j]);
    }
    int i;
    for (i = 0; i < num_threads; i++){
        pthread_join(thread_id[i], NULL);
        num_iter = worker_thread[i].num_iter;
    }
    // printf("THREADS DONE\n");
    // print_grid(src_grid);
    free(worker_thread);
    free(thread_id);
    free((void *) to_delete->element);
    free((void *) to_delete); 
    return num_iter;
}

/*
CHUNK ROWS [1,N-2]
converged= 0
init src and dest pointers
while (!converged){
    if (tid ==0)
        diff = 0
}
*/
void *jacobi_worker(void* args){
    thread_payload* data = (thread_payload*) args;
    
    double p_diff = 0;
    grid_t *tmp;
    float old, new;
    int i,j;
    

    while(!converged){
        pthread_barrier_wait(&barrier);
        if (data->tid ==0){
            // print_grid(data->src_grid);
            // printf("continue: %d\n",converged);
            diff = 0.0;
            num_elements = 0;
        }
        p_diff= 0;
        data->p_num_elements=0;
        pthread_barrier_wait(&barrier);


        if(data->tid<(data->num_threads-1)){
            // printf("%d\n",data->offset);
            for(i=data->offset;i<(data->offset+data->chunk_size);i++){
                for (j = 1; j < (data->src_grid->dim - 1); j++) {
                    old = data->src_grid->element[i * data->src_grid->dim + j];
                    new = 0.25 * (data->src_grid->element[(i - 1) * data->src_grid->dim + j] +\
                                data->src_grid->element[(i + 1) * data->src_grid->dim + j] +\
                                data->src_grid->element[i * data->src_grid->dim + (j + 1)] +\
                                data->src_grid->element[i * data->src_grid->dim + (j - 1)]);
                    data->dest_grid->element[i * data->dest_grid->dim + j] = new; /* Update the grid-point value. */
                    p_diff = p_diff + fabs(new - old); /* Calculate the difference in values. */
                    
                    data->p_num_elements++;
                }
            }
        }else{ /*last thread has to take extra elements*/
            // printf("final: %d\n",data->chunk_size);
            for(i=data->offset;i<(data->src_grid->dim-1);i++){
                for (j = 1; j < (data->src_grid->dim - 1); j++) {
                    old = data->src_grid->element[i * data->src_grid->dim + j];
                    new = 0.25 * (data->src_grid->element[(i - 1) * data->src_grid->dim + j] +\
                                data->src_grid->element[(i + 1) * data->src_grid->dim + j] +\
                                data->src_grid->element[i * data->src_grid->dim + (j + 1)] +\
                                data->src_grid->element[i * data->src_grid->dim + (j - 1)]);

                    data->dest_grid->element[i * data->dest_grid->dim + j] = new; /* Update the grid-point value. */
                    p_diff = p_diff + fabs(new - old); /* Calculate the difference in values. */
                    
                    data->p_num_elements++;            
                }           
            }
        }

        // update interior pts
        //calculate p_diff
        // printf("ID: %d. p_diff: %f\n",data->tid,p_diff);
        pthread_mutex_lock(&mutex);
        diff = diff + p_diff;
        num_elements = num_elements + data->p_num_elements;
        pthread_mutex_unlock(&mutex);
        pthread_barrier_wait(&barrier);
        data->num_iter++;
        // printf("diff: %f\n",diff);
        // printf("diff: %f\n",diff);
        if (diff/num_elements < data->eps){ 
            converged = 1;
        }
        tmp = data->src_grid;
        data->src_grid = data->dest_grid;
        data->dest_grid = tmp;
        // printf("continue: %d\n",converged);
        
    }
    
    pthread_exit(NULL); 
}

/* Create a grid with the specified initial conditions. */
grid_t* create_grid(int dim, float min, float max)
{
    grid_t *grid = (grid_t *)malloc (sizeof(grid_t));
    if (grid == NULL)
        return NULL;

    grid->dim = dim;
	fprintf(stderr, "Creating a grid of dimension %d x %d\n", grid->dim, grid->dim);
	grid->element = (float *) malloc(sizeof(float) * grid->dim * grid->dim);
    if (grid->element == NULL)
        return NULL;

    int i, j;
	for (i = 0; i < grid->dim; i++) {
		for (j = 0; j < grid->dim; j++) {
            grid->element[i * grid->dim + j] = 0.0; 			
		}
    }

    /* Initialize the north side, that is row 0, with temperature values. */ 
    srand((unsigned)time(NULL));
	float val;		
    for (j = 1; j < (grid->dim - 1); j++) {
        val =  min + (max - min) * rand ()/(float)RAND_MAX;
        grid->element[j] = val; 	
    }

    return grid;
}

/* Creates a new grid and copies over the contents of an existing grid into it. */
grid_t* copy_grid(grid_t *grid) 
{
    grid_t *new_grid = (grid_t *)malloc(sizeof(grid_t));
    if (new_grid == NULL)
        return NULL;

    new_grid->dim = grid->dim;
	new_grid->element = (float *)malloc(sizeof(float) * new_grid->dim * new_grid->dim);
    if (new_grid->element == NULL)
        return NULL;

    int i, j;
	for (i = 0; i < new_grid->dim; i++) {
		for (j = 0; j < new_grid->dim; j++) {
            new_grid->element[i * new_grid->dim + j] = grid->element[i * new_grid->dim + j] ; 			
		}
    }

    return new_grid;
}

/* Print grid to screen. */
void print_grid(grid_t *grid)
{   
    printf("PRINTING GRID\n");
    int i, j;
    for (i = 0; i < grid->dim; i++) {
        for (j = 0; j < grid->dim; j++) {
            printf("%f\t", grid->element[i * grid->dim + j]);
        }
        printf("\n");
    }
    printf("\n");
}


/* Print out statistics for the converged values of the interior grid points, including min, max, and average. */
void print_stats(grid_t *grid)
{
    float min = INFINITY;
    float max = 0.0;
    double sum = 0.0;
    int num_elem = 0;
    int i, j;

    for (i = 1; i < (grid->dim - 1); i++) {
        for (j = 1; j < (grid->dim - 1); j++) {
            sum += grid->element[i * grid->dim + j];

            if (grid->element[i * grid->dim + j] > max) 
                max = grid->element[i * grid->dim + j];

             if(grid->element[i * grid->dim + j] < min) 
                min = grid->element[i * grid->dim + j];
             
             num_elem++;
        }
    }
                    
    printf("AVG: %f\n", sum/num_elem);
	printf("MIN: %f\n", min);
	printf("MAX: %f\n", max);
	printf("\n");
}

/* Calculate the mean squared error between elements of two grids. */
double grid_mse(grid_t *grid_1, grid_t *grid_2)
{
    double mse = 0.0;
    int num_elem = grid_1->dim * grid_1->dim;
    int i;

    for (i = 0; i < num_elem; i++) 
        mse += (grid_1->element[i] - grid_2->element[i]) * (grid_1->element[i] - grid_2->element[i]);
                   
    return mse/num_elem; 
}



		

