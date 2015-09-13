#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"




int* dev_array;

namespace StreamCompaction {
	namespace Efficient {

		__global__ void kern_up_sweep(int n, int m_power, int* x) //m_power = 2^d
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index > 0 && index < n && ((index + 1) % (m_power * 2) == 0))
			{
				x[index] = x[index] + x[index - m_power];
			}
		}


		__global__ void kern_down_sweep(int n, int m_power, int* x) //m_power = 2^(log2(n)-1-d)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index>0 && index < n && ((index + 1) % (m_power * 2) == 0))
			{
				int tmp = x[index];
				x[index] = x[index] + x[index - m_power]; //sum
				x[index - m_power] = tmp; //swap
			}

		}

		__global__ void kern_set_value(int index,int val,int* x)
		{
			x[index] = val;
		}
		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {
			// up sweep

			int m_power = ilog2ceil(n);
			int new_n = pow(2, m_power);

			dim3 threadsPerBlock(blockSize);
			dim3 fullBlocksPerGrid((new_n + blockSize - 1) / blockSize);

			//init the array
			cudaMalloc((void**)&dev_array, new_n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_array1 failed!");

			cudaMemset(dev_array, 0, new_n*sizeof(int));
			checkCUDAErrorFn("cudaMemset dev_array failed!");
			
			cudaMemcpy(dev_array, idata, n*sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy dev_array failed!");

			//up sweep
			for (int d = 0; d < m_power; d++)
			{
				int pow_2_d = pow(2, d);
				kern_up_sweep << <fullBlocksPerGrid, threadsPerBlock >> >(new_n, pow_2_d, dev_array);
			}

			//down sweep
			
			kern_set_value << <1, 1 >> > (new_n - 1, 0, dev_array); //insert 0
			
			
			for (int d = 0; d < m_power; d++)
			{
				int nn = m_power - 1 - d;
				int pow_2_log2n_minus_d = pow(2, nn);

				kern_down_sweep << <fullBlocksPerGrid, threadsPerBlock >> >(new_n, pow_2_log2n_minus_d, dev_array);
			}

			//copy data
			cudaMemcpy(odata, dev_array, n*sizeof(int), cudaMemcpyDeviceToHost);

		}






		/**
		 * Performs stream compaction on idata, storing the result into odata.
		 * All zeroes are discarded.
		 *
		 * @param n      The number of elements in idata.
		 * @param odata  The array into which to store elements.
		 * @param idata  The array of elements to compact.
		 * @returns      The number of elements remaining after compaction.
		 */
		int compact(int n, int *odata, const int *idata) {
			
			dim3 threadsPerBlock(blockSize);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			//copy data to device
			int* dev_idata;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_idata failed!");

			cudaMemset(dev_idata, 0, n*sizeof(int));
			checkCUDAErrorFn("cudaMemset dev_idata failed!");

			cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy dev_idata failed!");

			
			// map the idata to bools
			int* dev_bools;
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_bools failed!");

			cudaMemset(dev_bools, 0, n*sizeof(int));
			checkCUDAErrorFn("cudaMemset dev_bools failed!");

			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid ,threadsPerBlock>> >(n, dev_bools, dev_idata);

			//scan the bools to get the indices
			

			int* host_bools = new int[n];
			cudaMemcpy(host_bools, dev_bools, n*sizeof(int), cudaMemcpyDeviceToHost);
			int* host_indices = new int[n];
			
			scan(n, host_indices, host_bools);  //input is host data

			int* dev_indices;
			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_indices failed!");

			cudaMemcpy(dev_indices, host_indices, n*sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy dev_indices failed!");

			//run scatter
			int* dev_odata;
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_bools failed!");

			cudaMemset(dev_odata, 0, n*sizeof(int));
			checkCUDAErrorFn("cudaMemset dev_bools failed!");
			
			StreamCompaction::Common::kernScatter << < fullBlocksPerGrid, threadsPerBlock >> > (n, dev_odata,dev_idata,dev_bools,dev_indices);
			
			//copy back to host
			cudaMemcpy(odata, dev_odata, n*sizeof(int), cudaMemcpyDeviceToHost);

			return host_indices[n - 1]; //num of non-zero
			

			
		}


	}
}

