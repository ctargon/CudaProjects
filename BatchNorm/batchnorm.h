#include <cuda_runtime.h> 
#include <cuda.h> 
#include "utils.h"

/*
	launcher function for the batchnorm kernels
*/ 

void launch_batchnorm(float *batch, float *mean, float *var, 
					  size_t batch_size, size_t rows, size_t cols, 
					  size_t depth);
