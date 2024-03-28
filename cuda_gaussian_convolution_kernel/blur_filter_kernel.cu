/* Blur filter. Device code. */

#ifndef _BLUR_FILTER_KERNEL_H_
#define _BLUR_FILTER_KERNEL_H_

#include "blur_filter.h"

__global__ void blur_filter_kernel (const float *in, float *out, int size)
{
    /* Obtain thread index within the thread block */
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;

	/* Obtain block index within the grid */
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

    /* Find position in matrix */
	int column_number = blockDim.x * blockX + threadX;
	int row_number = blockDim.y * blockY + threadY;

    int i,j;
    float blur_val = 0;
    int num_neighbors = 0;

    int row;
    int col;

    for(i=-BLUR_SIZE; i<(BLUR_SIZE+1);i++){
        for(j = -BLUR_SIZE; j<(BLUR_SIZE+1);j++){
            row = row_number+i;
            col = column_number +j;
            if((row > -1) && (col > -1) && (row < size) && (col < size)){
                blur_val += in[row*size + col];
                num_neighbors++;
            }
        }
    }

    out[row_number*size + column_number] = blur_val/num_neighbors;
    return;
}

#endif /* _BLUR_FILTER_KERNEL_H_ */
