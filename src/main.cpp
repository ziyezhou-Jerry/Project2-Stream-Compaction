/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radixsort.h>
#include <algorithm>
#include <vector>
#include <ctime>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "testing_helpers.hpp"


#define M_SIZE 2000


//help function
void tick()
{

}

void tock()
{

}



int main(int argc, char* argv[]) {
   
	//for (int m_i = 8; m_i < 16; m_i++)
	//{

		int SIZE = 1 << 8;
		const int NPOT = SIZE - 3;
		int* a = new int[SIZE];
		int*b = new int[SIZE]; 
		int*c = new int[SIZE];



		int m_array[M_SIZE];
		int m_out[M_SIZE];
		for (int i = 0; i < M_SIZE / 2; i++)
		{
			m_array[i] = M_SIZE / 2 - i;
		}
		for (int i = M_SIZE / 2; i < M_SIZE; i++)
		{
			m_array[i] = i - M_SIZE / 2;
		}

		std::vector<int> m_array_vec(m_array, m_array + M_SIZE);


		//cuda event init
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		float milliseconds = 0;


		// === Sort Test ===
		printf("\n");
		printf("*****************************\n");
		printf("** Radix Sort TESTS **\n");
		printf("*****************************\n");
		printDesc("radix sort using GPU");



		StreamCompaction::RadixSort::radixsort(M_SIZE, m_out, m_array);



		printDesc("sort using CPU");


		cudaEventRecord(start);
		std::sort(m_array_vec.begin(), m_array_vec.end());
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		std::cout << "printf: " << milliseconds << '\n';




		printArray(60, m_out, true);
		printArray(60, &m_array_vec[0], true);

		//  === Scan tests ====

		printf("\n");
		printf("****************\n");
		printf("** SCAN TESTS **\n");
		printf("****************\n");

		genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
		a[SIZE - 1] = 0;
		printArray(SIZE, a, true);

		zeroArray(SIZE, b);
		printDesc("cpu scan, power-of-two");
		StreamCompaction::CPU::scan(SIZE, b, a);
		//printArray(SIZE, b, true);

		zeroArray(SIZE, c);
		printDesc("cpu scan, non-power-of-two");
		StreamCompaction::CPU::scan(NPOT, c, a);
		//printArray(NPOT, b, true);
		printCmpResult(NPOT, b, c);

		zeroArray(SIZE, c);
		printDesc("naive scan, power-of-two");
		StreamCompaction::Naive::scan(SIZE, c, a);
		//printArray(SIZE, c, true);
		printCmpResult(SIZE, b, c);

		zeroArray(SIZE, c);
		printDesc("naive scan, non-power-of-two");
		StreamCompaction::Naive::scan(NPOT, c, a);
		//printArray(SIZE, c, true);
		printCmpResult(NPOT, b, c);


		zeroArray(SIZE, c);
		printDesc("work-efficient scan, non-power-of-two");
		StreamCompaction::Efficient::scan(NPOT, c, a);
		//printArray(NPOT, c, true);
		printCmpResult(NPOT, b, c);


		zeroArray(SIZE, c);
		printDesc("work-efficient scan, power-of-two");
		StreamCompaction::Efficient::scan(SIZE, c, a);
		//printArray(SIZE, c, true);
		printCmpResult(SIZE, b, c);


		zeroArray(SIZE, c);
		printDesc("work-efficient share_mem scan, power-of-two");
		StreamCompaction::Efficient::scan_share_mem(SIZE, c, a);
		/*printArray(SIZE, a, true);
		printArray(SIZE, c, true);
		printArray(SIZE, b, true);*/
		printCmpResult(SIZE, b, c);


		zeroArray(SIZE, c);
		printDesc("work-efficient share_mem scan, non-power-of-two");
		StreamCompaction::Efficient::scan_share_mem(NPOT, c, a);
		printCmpResult(NPOT, b, c);


		zeroArray(SIZE, c);
		printDesc("thrust scan, power-of-two");
		StreamCompaction::Thrust::scan(SIZE, c, a);
		//printArray(SIZE, c, true);
		printCmpResult(SIZE, b, c);

		zeroArray(SIZE, c);
		printDesc("thrust scan, non-power-of-two");
		StreamCompaction::Thrust::scan(NPOT, c, a);
		//printArray(NPOT, c, true);
		printCmpResult(NPOT, b, c);
	//}

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    expectedCount = count;
   // printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    expectedNPOT = count;
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

	zeroArray(SIZE, c);
	printDesc("work-efficient share mem compact, power-of-two");
	count = StreamCompaction::Efficient::compact_share_mem(SIZE, c, a);
	//printArray(count, c, true);
	printCmpLenResult(count, expectedCount, b, c);

	zeroArray(SIZE, c);
	printDesc("work-efficient share mem compact, non-power-of-two");
	count = StreamCompaction::Efficient::compact_share_mem(NPOT, c, a);
	//printArray(count, c, true);
	printCmpLenResult(count, expectedNPOT, b, c);


	
	//system("pause");
}
