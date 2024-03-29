CUDA_PATH ?= /usr/local/cuda-10.1
HOST_COMPILER ?= g++
NVCC = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER) 
NVCCFLAGS	:= -m64 -O3 -gencode arch=compute_75,code=compute_75
SIZE = 8192
TILE_SIZE = 32

all: blur_filter

blur_filter: blur_filter.cu blur_filter_kernel.cu blur_filter_gold.cpp 
	$(NVCC) -o blur_filter blur_filter.cu blur_filter_gold.cpp $(NVCCFLAGS)

clean:
	rm blur_filter 

run:
	./blur_filter $(SIZE) $(TILE_SIZE)

empty:
	clear

redo: empty all run