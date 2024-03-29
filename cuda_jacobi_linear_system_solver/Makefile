CUDA_PATH ?= /usr/local/cuda-10.1
HOST_COMPILER ?= g++
NVCC = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER) 
NVCCFLAGS	:= -m64 -O3 -gencode arch=compute_75,code=compute_75
THREAD_BLOCk = 32
MATRIX_SIZE = 2048

all: jacobi_iteration

jacobi_iteration: jacobi_iteration.cu jacobi_iteration_kernel.cu jacobi_iteration_gold.cpp 
	$(NVCC) -o jacobi_iteration jacobi_iteration.cu jacobi_iteration_gold.cpp $(NVCCFLAGS)

clean:
	rm jacobi_iteration 

run:
	./jacobi_iteration $(THREAD_BLOCk) $(MATRIX_SIZE)

clear:
	clear

redo: clear clean all run


