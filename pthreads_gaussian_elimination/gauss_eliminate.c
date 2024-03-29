/* Gaussian elimination code.
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -std=c99 -O3 -Wall -lpthread -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix,int);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix*);
float get_random_number(int, int);
int check_results(float *, float *, int, float);
void* worker_gauss(void*);

typedef struct thread_payload{
    Matrix* U;
    int tid;
    int num_threads;
    unsigned int num_columns;
    unsigned int num_rows;
    int row_chunk_size;
    int row_offset;
    int col_chunk_size;
    int col_offset;
} thread_payload;

pthread_barrier_t barrier;  
pthread_mutex_t mutex;

float scalar;


int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s matrix-size num_threads\n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        fprintf(stderr, "num_threads: number of threads\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    Matrix A;			                                            /* Input matrix */
    Matrix U_reference;		                                        /* Upper triangular matrix computed by reference code */
    Matrix U_mt;			                                        /* Upper triangular matrix computed by pthreads */

    fprintf(stderr, "Generating input matrices\n");
    srand(time (NULL));                                             /* Seed random number generator */
    A = allocate_matrix(matrix_size, matrix_size, 1);               /* Allocate and populate random square matrix */
    U_reference = allocate_matrix (matrix_size, matrix_size, 0);    /* Allocate space for reference result */
    U_mt = allocate_matrix (matrix_size, matrix_size, 0);           /* Allocate space for multi-threaded result */

    /* Copy contents A matrix into U matrices */
    int i, j;
    for (i = 0; i < A.num_rows; i++) {
        for (j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    
    int status = compute_gold(U_reference.elements, A.num_rows);
  
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    if (status < 0) {
        fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
        exit(EXIT_FAILURE);
    }
  
    status = perform_simple_check(U_reference);	/* Check that principal diagonal elements are 1 */ 
    if (status < 0) {
        fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");
  
    /* FIXME: Perform Gaussian elimination using pthreads. 
     * The resulting upper triangular matrix should be returned in U_mt */
    fprintf(stderr, "\nPerforming gaussian elimination using pthreads\n");
    gettimeofday(&start, NULL);
    gauss_eliminate_using_pthreads(U_mt,num_threads);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    /* Check if pthread result matches reference solution within specified tolerance */
    fprintf(stderr, "\nChecking results\n");
    int size = matrix_size * matrix_size;
    int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);
    fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");

    /* Free memory allocated for matrices */
    free(A.elements);
    free(U_reference.elements);
    free(U_mt.elements);

    exit(EXIT_SUCCESS);
}


/* FIXME: Write code to perform gaussian elimination using pthreads */
void gauss_eliminate_using_pthreads(Matrix U,int num_threads){

    pthread_t* thread_id = malloc(num_threads*sizeof(pthread_t));
    thread_payload *thread_data = malloc(num_threads * sizeof(thread_payload));
    if ((thread_data == NULL || (thread_id == NULL))){
        fprintf(stderr,"malloc failed when making thread args\n");
        exit(EXIT_FAILURE);
    }

    pthread_mutex_init(&mutex, NULL);
    pthread_barrier_init(&barrier, NULL, num_threads);
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);

    int j;
    // print_matrix(&U);
    for (j = 0; j < num_threads; j++){
        //TDOD
        thread_data[j].U = &U;
        thread_data[j].tid = j;
        thread_data[j].num_threads =num_threads;
        thread_data[j].row_chunk_size = 0;
        thread_data[j].row_offset = 0;
        thread_data[j].col_chunk_size = 0;
        thread_data[j].col_offset = 0;
        pthread_create(&thread_id[j],&attributes,worker_gauss,(void*) &thread_data[j]);
    }
    int i;
    for (i = 0; i < num_threads; i++){
        pthread_join(thread_id[i], NULL);
    }
    free(thread_data);
    free(thread_id);
}

void* worker_gauss(void* args){
    thread_payload* data = (thread_payload*)args;
    int col,row;
    //DIVISION STEP
    for(row=0;row<data->U->num_rows;row++){
        if(data->tid == 0){
            scalar = data->U->elements[row*data->U->num_columns+row];
        }
        data->row_chunk_size = floor((data->U->num_columns-row)/data->num_threads);
        data->row_offset = data->tid * data->row_chunk_size;
        pthread_barrier_wait(&barrier);
        
        if(data->tid<(data->num_threads-1)){
            for(col = data->row_offset;col<(data->row_offset+data->row_chunk_size);col++){
                if (data->U->elements[(data->U->num_rows*row)+col])
                    data->U->elements[(data->U->num_rows*row)+col] = (float)(data->U->elements[(data->U->num_rows*row)+col] / scalar);
            }
        }else{//last thread clean up
            for(col = data->row_offset;col<(data->U->num_columns);col++){
                if (data->U->elements[(data->U->num_rows*row)+col])
                    data->U->elements[(data->U->num_rows*row)+col] = (float)(data->U->elements[(data->U->num_rows*row)+col] / scalar);
            }
        }
        pthread_barrier_wait(&barrier);
        //Elimination step
        data->col_chunk_size = floor((data->U->num_rows-row-1)/data->num_threads);
        data->col_offset = data->tid * data->col_chunk_size;
        int r,c;
        float r_scale;
        if(data->tid<(data->num_threads-1)){
            for(r=data->col_offset+1+row;r<(data->col_offset+1+row+data->col_chunk_size);r++){
                r_scale = data->U->elements[(data->U->num_rows*r)+row];
                for(c=row;c<(data->U->num_columns);c++){
                    data->U->elements[(data->U->num_rows*r)+c] = data->U->elements[(data->U->num_rows*r)+c] - (data->U->elements[(data->U->num_rows*row)+c]*r_scale);
                }
            }
        }else{//last thread
            for(r=data->col_offset+1+row;r<(data->U->num_rows);r++){
                r_scale = data->U->elements[(data->U->num_rows*r)+row];
                for(c=row;c<(data->U->num_columns);c++){
                    data->U->elements[(data->U->num_rows*r)+c] = data->U->elements[(data->U->num_rows*r)+c] - (data->U->elements[(data->U->num_rows*row)+c]*r_scale);
                }
            }
        }
        pthread_barrier_wait(&barrier);
        if(data->tid == 0){
            // printf("good? %d\n",perform_simple_check(*(data->U)));
            // printf("after\n");
            // print_matrix(data->U);
        }
    }
    

    //Go Eliminate


    pthread_exit(NULL); 
}

/* Check if results generated by single threaded and multi threaded versions match within tolerance */
int check_results(float *A, float *B, int size, float tolerance)
{
    int i;
    for (i = 0; i < size; i++)
        if(fabsf(A[i] - B[i]) > tolerance)
            return -1;
    return 0;
}

void print_matrix(const Matrix *mat){
    printf("PRINTING GRID\n");
    int i, j;
    for (i = 0; i < mat->num_columns; i++) {
        for (j = 0; j < mat->num_rows; j++) {
            printf("%f\t", mat->elements[i * mat->num_rows + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    Matrix M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));
  
    for (i = 0; i < size; i++) {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }
  
    return M;
}

/* Return a random floating-point number between [min, max] */ 
float get_random_number(int min, int max)
{
    return (float)floor((double)(min + (max - min + 1) * ((float)rand() / (float)RAND_MAX)));
}

/* Perform simple check on upper triangular matrix if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
    int i;
    for (i = 0; i < M.num_rows; i++)
        if ((fabs(M.elements[M.num_rows * i + i] - 1.0)) > 1e-6)
            return -1;
  
    return 0;
}
