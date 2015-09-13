#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "radixsort.h"
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
	namespace RadixSort {


		__global__ kern_get_k_bit_array(int n, int k, int* odata, const int* idata)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index < n)
			{
				odata[index] = (idata[index] & (1 << k)) >> k; //get the kth bit of the cur int
			}
		}

		__global__ kern_inv_array(int n, int* odata, const int* idata) //1-->0  0-->1
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index < n)
			{
				odata[index] = std::abs(idata[index]-1);
			}
		}




		void radixsort(int n, int *odata, const int *idata) //assume that all the bits are 32 bits
		{
			
			
			
			dim3 threadsPerBlock(blockSize);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			
			int * dev_b_array;
			cudaMalloc((void**)&dev_b_array, n*sizeof(int));
			int * dev_e_array;
			cudaMalloc((void**)&dev_e_array, n*sizeof(int));
			int * dev_f_array;
			cudaMalloc((void**)&dev_f_array, n*sizeof(int));
			int * dev_t_array;
			cudaMalloc((void**)&dev_t_array, n*sizeof(int));

			int * dev_idata;
			cudaMalloc((void**)&dev_idata, n*sizeof(int));

			int * dev_odata;
			cudaMalloc((void**)&dev_odata, n*sizeof(int));

			
			cudaMemcpy(dev_idata,idata,n*sizeof(int),cudaMemcpyHostToDevice);

			for (int k = 0; k<32; k++)
			{
				//get b array
				kern_get_k_bit_array << <fullBlocksPerGrid, threadsPerBlock >> > (n, k, dev_b_array,dev_idata);

				//get e array

			}
		}

	
	}
}