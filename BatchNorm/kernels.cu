#include "batchnorm.h" 
#include <math.h>

#define BLOCK_SIZE 1024
#define BLOCK2D 32
#define EPSILON 1e-8
/*
	Kernel to calculate the mean
 */
__global__ 
void mean_k(float *batch, float *mean, size_t batch_size, 
		  size_t rows, size_t cols, size_t depth)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < rows * cols * depth)
	{
		mean[idx] = 0;
		for (size_t b = 0; b < batch_size; b++)
		{
			mean[idx] += batch[(b * rows * cols * depth) + idx];
		}
	}
	__syncthreads();
	if (idx < rows * cols * depth) mean[idx] /= batch_size;
} 

/*
	Kernel to calculate the variance
 */
__global__ 
void variance_k(float *batch, float *mean, float *var, size_t batch_size, 
		  size_t rows, size_t cols, size_t depth)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < rows * cols * depth)
	{
		var[idx] = 0;
		for (size_t b = 0; b < batch_size; b++)
		{
			var[idx] += pow(batch[(b * rows * cols * depth) + idx] - mean[idx], 2);
		}
	}
	__syncthreads();
	if (idx < rows * cols * depth) var[idx] /= batch_size;
} 

/*
	Kernel to calculate batch normalization
 */
__global__ 
void compute(float *batch, float *mean, float *var, size_t batch_size, 
		  size_t rows, size_t cols, size_t depth, float gamma,
		  float beta)
{
	int batch_i = blockIdx.x * blockDim.x + threadIdx.x;
	int feature_i = blockIdx.y * blockDim.y + threadIdx.y;

	if (batch_i < batch_size && feature_i < rows * cols * depth)
	{
		float prev_val = batch[(batch_i * rows * cols * depth) + feature_i];
		float mu = mean[feature_i];
		float sig = var[feature_i];
		float norm = (prev_val - mu) / sqrt(sig + EPSILON);
		float scaled = (gamma * norm) + beta;
		batch[(batch_i * rows * cols * depth) + feature_i] = scaled;
	}
} 



void launch_batchnorm(float *batch, float *mean, float *var, 
					  size_t batch_size, size_t rows, size_t cols, 
					  size_t depth)
{
	int n_features = rows * cols * depth;
	dim3 blockSize(BLOCK_SIZE, 1, 1);
	dim3 gridSize(ceil(n_features / (float)BLOCK_SIZE), 1, 1);

	mean_k<<<gridSize, blockSize>>>(batch, mean, batch_size, rows, cols, depth); 
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());  

	variance_k<<<gridSize, blockSize>>>(batch, mean, var, batch_size, rows, cols, depth); 
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());  

	dim3 computeBlockSize(BLOCK2D, BLOCK2D, 1);
	dim3 computeGridSize(ceil(batch_size / (float)BLOCK2D), ceil(n_features / (float)BLOCK2D), 1);

	compute<<<computeGridSize, computeBlockSize>>>(batch, mean, var, batch_size, rows, cols, depth, 1.0, 0.0); 
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());  

}




