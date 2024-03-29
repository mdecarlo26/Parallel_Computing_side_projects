/* Reference code implementing the box blur filter.

    Build and execute as follows: 
        make clean && make 
        ./blur_filter size

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "cuda_runtime.h"
// #include "device_launch_parameters.h"

// #define TILE_SIZE 32

/* #define DEBUG */

/* Include the kernel code */
#include "blur_filter_kernel.cu"

extern "C" void compute_gold(const image_t, image_t);
void compute_on_device(const image_t, image_t,int t);
int check_results(const float *, const float *, int, float);
void print_image(const image_t);

//helper funcs for GPU work
void copy_image_to_device(image_t);
void copy_image_from_device(image_t,image_t);
void free_image_on_device(image_t*);
void gauss_filter_on_device();
image_t allocate_image_on_device(image_t);
void check_CUDA_error(const char *);


int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s size tile_size\n", argv[0]);
        fprintf(stderr, "size: Height of the image. The program assumes size x size image.\n");
        fprintf(stderr, "tile_size: desired thread block size. MAX IS 32\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate memory for the input and output images */
    int size = atoi(argv[1]);
    int tile_size = atoi(argv[2]);

    fprintf(stderr, "Creating %d x %d images\n", size, size);
    image_t in, out_gold, out_gpu;
    in.size = out_gold.size = out_gpu.size = size;
    in.element = (float *)malloc(sizeof(float) * size * size);
    out_gold.element = (float *)malloc(sizeof(float) * size * size);
    out_gpu.element = (float *)malloc(sizeof(float) * size * size);
    if ((in.element == NULL) || (out_gold.element == NULL) || (out_gpu.element == NULL)) {
        perror("Malloc");
        exit(EXIT_FAILURE);
    }

    /* Poplulate our image with random values between [-0.5 +0.5] */
    srand(time(NULL));
    int i;
    for (i = 0; i < size * size; i++)
        in.element[i] = rand()/(float)RAND_MAX -  0.5;
  
   /* Calculate the blur on the CPU. The result is stored in out_gold. */
    fprintf(stderr, "Calculating blur on the CPU\n"); 
    struct timeval start, stop;
    gettimeofday(&start, NULL);

    compute_gold(in, out_gold); 

    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.6f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

#ifdef DEBUG 
   print_image(in);
   print_image(out_gold);
#endif

   /* FIXME: Calculate the blur on the GPU. The result is stored in out_gpu. */
   fprintf(stderr, "Calculating blur on the GPU\n");
//    gettimeofday(&start, NULL);

   compute_on_device(in, out_gpu,tile_size);

    // gettimeofday(&stop, NULL);
    // fprintf(stderr, "GPU run time = %0.6f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

   /* Check CPU and GPU results for correctness */
   fprintf(stderr, "Checking CPU and GPU results\n");
   int num_elements = out_gold.size * out_gold.size;
   float eps = 1e-6;    /* Do not change */
   int check;
   check = check_results(out_gold.element, out_gpu.element, num_elements, eps);
   if (check == 0) 
       fprintf(stderr, "TEST PASSED\n");
   else
       fprintf(stderr, "TEST FAILED\n");
   
   /* Free data structures on the host */
   free((void *)in.element);
   free((void *)out_gold.element);
   free((void *)out_gpu.element);

    exit(EXIT_SUCCESS);
}

image_t allocate_image_on_device(image_t im){
    image_t im_device =im;
    int im_size_bytes = im.size * im.size *sizeof(float);

    cudaMalloc((void**)&im_device.element,im_size_bytes);
    if(im_device.element == NULL){//error checking
        fprintf(stderr,"CudaMalloc failed\n");
        exit(EXIT_FAILURE);
    }
    return im_device;
}

void free_image_on_device(image_t* im){
    cudaFree(im->element);
    // im->element == NULL;
}

void copy_image_to_device(image_t im_device,image_t im_host){
    int im_size_bytes = im_host.size*im_host.size*sizeof(float);

    cudaMemcpy(im_device.element,im_host.element,im_size_bytes,cudaMemcpyHostToDevice);
}

void copy_image_from_device(image_t host,image_t device){
    int size_bytes = host.size*host.size*sizeof(float);
    cudaMemcpy(host.element,device.element,size_bytes,cudaMemcpyDeviceToHost);
}

/* FIXME: Complete this function to calculate the blur on the GPU */
void compute_on_device(const image_t in, image_t out,int tile_size)
{
    struct timeval start, stop,first,last;
    gettimeofday(&start, NULL);
    gettimeofday(&first, NULL);

    image_t in_on_device = allocate_image_on_device(in);
    copy_image_to_device(in_on_device,in);
    image_t out_on_device = allocate_image_on_device(out);

    gettimeofday(&stop, NULL);
    fprintf(stderr, "Data setup time = %0.6f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    //Copy input image to GPU global mem
    // cudaMalloc((void**)&in_on_device)

    dim3 threads(tile_size,tile_size);
    fprintf(stderr,"Setting up a %d x %d grid of thread blocks (%d,%d)\n", (in.size + tile_size -1)/tile_size,\\
        (in.size + tile_size -1)/tile_size,tile_size,tile_size);
    dim3 grid((in.size + tile_size -1)/tile_size,(in.size + tile_size -1)/tile_size);

    gettimeofday(&start, NULL);

    blur_filter_kernel<<< grid,threads >>>(in_on_device.element,out_on_device.element,in_on_device.size);
    cudaDeviceSynchronize();
    gettimeofday(&stop, NULL);
    fprintf(stderr, "GPU compute time = %0.6f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    check_CUDA_error("Error in kernel");

    copy_image_from_device(out,out_on_device);

    gettimeofday(&start, NULL);
    free_image_on_device(&in_on_device);
    free_image_on_device(&out_on_device);
    gettimeofday(&stop, NULL);
    gettimeofday(&last, NULL);
    fprintf(stderr, "Free Data time = %0.6f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    fprintf(stderr, "Overall GPU time (with data transfer) = %0.6f s\n", (float)(last.tv_sec - first.tv_sec\
                + (last.tv_usec - first.tv_usec) / (float)1000000));
    return;
}

void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
}
/* Check correctness of results */
int check_results(const float *pix1, const float *pix2, int num_elements, float eps) 
{
    int i;
    for (i = 0; i < num_elements; i++)
        if (fabsf((pix1[i] - pix2[i])/pix1[i]) > eps) 
            return -1;
    
    return 0;
}

/* Print out the image contents */
void print_image(const image_t img)
{
    int i, j;
    float val;
    for (i = 0; i < img.size; i++) {
        for (j = 0; j < img.size; j++) {
            val = img.element[i * img.size + j];
            printf("%0.4f ", val);
        }
        printf("\n");
    }

    printf("\n");
}
