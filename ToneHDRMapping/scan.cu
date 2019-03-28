#include "scan.h"
#include <math.h>



/*
 

 
 */
__global__ 
void naive_scan(float *d_in, float *d_out, size_t length)
{
	/*
		Your kernel here: Make sure to check for boundary conditions
	*/
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int j;

	__shared__ float tmp[BLOCK];

	if (idx < length) tmp[threadIdx.x] = d_in[idx];

	for (j = 1; j <= threadIdx.x; j *= 2)
	{
		__syncthreads();
		float in1 = tmp[threadIdx.x - j];
		__syncthreads();
		tmp[threadIdx.x] += in1;
	}

	if (idx < length) d_out[idx] = tmp[threadIdx.x];
}

__global__
void naive_scan2(float *d_in, float *d_out, size_t length)
{
	int i = threadIdx.x;
	int n = blockDim.x;

	for (int offset = 1; offset < n; offset *= 2)
	{
		float t;
		if (i >= offset) t = d_in[i - offset];
		__syncthreads();
		if (i >= offset) d_in[i] += t;
		__syncthreads(); 
	}
	if (i < length) d_out[i] = d_in[i];
}

__global__
void scan(float *d_in, float *d_out, float *d_sums, size_t length)
{
	__shared__ float temp[2 * BLOCK];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// load shared memory
	if (idx < length / 2) temp[2 * threadIdx.x] = d_in[2 * idx];
	else temp[2 * threadIdx.x] = 0;
	if (2 * idx + 1 < length) temp[2 * threadIdx.x + 1] = d_in[2 * idx + 1];
	else temp[2 * threadIdx.x + 1] = 0;
	__syncthreads();

	for (int stride = 1; stride <= BLOCK; stride *= 2)
	{
		int i = (threadIdx.x + 1) * stride * 2 - 1; // data index
		if (i < 2 * BLOCK)
		{
			temp[i] += temp[i - stride];
		}
		__syncthreads();
	}

	for (int stride = BLOCK / 2; stride > 0; stride /= 2)
	{
		int i = (threadIdx.x + 1) * stride * 2 - 1;
		if (i + stride < 2 * BLOCK)
		{
			temp[i + stride] += temp[i];
		}
		__syncthreads();
	}

	if (idx < length / 2) 
	{
		d_out[2 * idx] = temp[2 * threadIdx.x];
		d_out[2 * idx + 1] = temp[2 * threadIdx.x + 1];
	}

	if (d_sums && threadIdx.x == 0) d_sums[blockIdx.x] = temp[2 * BLOCK - 1];
}

__global__
void increment_scan(float *d_in, float *d_out, float *d_inc, size_t length)
{
	__shared__ float temp[2 * BLOCK];
	__shared__ float increment;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// load shared memory
	temp[2 * threadIdx.x] = d_out[2 * idx];
	temp[2 * threadIdx.x + 1] = d_out[2 * idx + 1];

	if (!threadIdx.x)
	{
		if (blockIdx.x == 0) increment = 0;
		else increment = d_inc[blockIdx.x - 1];
	}
	__syncthreads();

	temp[2 * threadIdx.x] += increment;
	temp[2 * threadIdx.x + 1] += increment; 
	__syncthreads();

	d_out[2 * idx] = temp[2 * threadIdx.x];
	d_out[2 * idx + 1] = temp[2 * threadIdx.x + 1];
}


void launch_scan(float *d_in, float *d_out, float *d_sums, float *d_incs, size_t length)
{
		// configure launch params here 
		dim3 block(BLOCK, 1, 1);
		int grid_d = ceil(length / (2.0 * BLOCK));
		dim3 grid(grid_d, 1, 1);

		scan<<<grid,block>>>(d_in, d_out, d_sums, length);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		int grid_d2 = ceil(grid_d / (2.0 * BLOCK));
		dim3 grid2(grid_d2, 1, 1);
		scan<<<grid2,block>>>(d_sums, d_incs, NULL, length);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		increment_scan<<<grid,block>>>(d_in, d_out, d_incs, length);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
}





