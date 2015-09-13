#pragma once

namespace StreamCompaction {
	namespace RadixSort {
		
		__global__ kern_get_k_bit_array(int n, int k, int* odata, int* idata);
		void radixsort(int n, int *odata, const int *idata);
	}
}