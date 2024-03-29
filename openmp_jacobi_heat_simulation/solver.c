/* Code for the Jacbi equation solver. 
 *
 * Compile as follows:
 * gcc -o solver solver.c solver_gold.c -fopenmp -O3 -Wall -std=c99 -lm
 *
 * If you wish to see debug info add the -D DEBUG option when compiling the code.
 */

// System includes
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>
#include "grid.h" 

extern int compute_gold_gauss(grid_t *);
extern int compute_gold_jacobi(grid_t *);
int compute_using_omp_jacobi(grid_t *, int);
void compute_grid_differences(grid_t *, grid_t *);
grid_t *create_grid(int, float, float);
grid_t *copy_grid(grid_t *);
void print_grid(grid_t *);
void print_stats(grid_t *);
double grid_mse(grid_t *, grid_t *);


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
	num_iter = compute_using_omp_jacobi(grid_3, num_threads);
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
int compute_using_omp_jacobi(grid_t *grid, int thread_count)
{		
    int num_iter = 0;
	int done = 0;
    int i, j;
	double diff;
	float old, new;
    int num_elements;
    float eps = 1e-4;

    grid_t *src_grid = grid;
    grid_t *dest_grid = copy_grid(grid);
    grid_t *to_delete= dest_grid;
    grid_t *tmp;
#   pragma omp parallel num_threads(thread_count) shared(diff,num_elements,eps,num_iter,src_grid,dest_grid,tmp,done) default(none) private(i,j,old,new)
    {
    while (!done){
#   pragma omp single
    {
        diff =0.0;
        num_elements = 0;
        num_iter++;
    }//implicit barrier
# pragma omp for reduction(+:diff,num_elements) collapse(2) 
    for (i = 1; i < (src_grid->dim - 1); i++) {
            for (j = 1; j < (src_grid->dim - 1); j++) {
                old = src_grid->element[i * src_grid->dim + j]; /* Store old value of grid point. */
                /* Apply the update rule. */	
                new = 0.25 * (src_grid->element[(i - 1) * src_grid->dim + j] +\
                              src_grid->element[(i + 1) * src_grid->dim + j] +\
                              src_grid->element[i * src_grid->dim + (j + 1)] +\
                              src_grid->element[i * src_grid->dim + (j - 1)]);

                dest_grid->element[i * dest_grid->dim + j] = new; /* Update the grid-point value. */
                diff +=  fabs(new - old); /* Calculate the difference in values. */
                num_elements+=1;
            }
        }
    //implicit barrier
#   pragma omp single
    {
    diff = diff/num_elements;
    if (diff < eps) 
        done = 1;
    }//implicit barrier
# pragma omp master
{
        tmp = src_grid;
        src_grid = dest_grid;
        dest_grid = tmp;
}
    }
    }//implicit barrier
    free((void *) to_delete->element);
    free((void *) to_delete); 
    return num_iter;
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



		

