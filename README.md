CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Ziye Zhou
* Tested on: Windows 8.1, i7-4910 @ 2.90GHz 32GB, GTX 880M 8192MB (Alienware)

### Description
1. Implemented CPU Scan & Stream Compaction
2. Implemented Naive GPU Scan & Stream Compaction
3. Implemented Work-Efficient GPU Scan & Stream Compaction
4. Tested Thrust's Implementation
5. Implemented Radix Sort (Extra Credit)
6. Compared the performance of CPU & GPU on Scan Algorithm

### Questions

* Compare all of these GPU Scan implementations (Naive, Work-Efficient, and
  Thrust) to the serial CPU version of Scan. Plot a graph of the comparison
  (with array size on the independent axis).
 ![alt tag](https://github.com/ziyezhou-Jerry/Project2-Stream-Compaction/blob/master/proj2_compare.png?raw=true)
![alt tag](https://github.com/ziyezhou-Jerry/Project2-Stream-Compaction/blob/master/proj2_thrust_running.png?raw=true)

* Write a brief explanation of the phenomena you see here.
  * Can you find the performance bottlenecks? Is it memory I/O? Computation? Is
    it different for each implementation?
From the performance analysis given below, I think the bootleneck is the memory I/O part. As we can see from the timeline, the computation time only takes a little portion of the whole time, but the memory I/O takes a lot. This is nearly the same for all implementations.

![alt tag](https://github.com/ziyezhou-Jerry/Project2-Stream-Compaction/blob/master/proj2_bottleneck_analysis.png?raw=true)

* Paste the output of the test program into a triple-backtick block in your
  README.

  Apart from the given test cases, I have also added my own test case for the radix sort. I used the STL sort method to get the CPU version of sort result and compare it with the GPU radix sort.
  <<<![alt tag](https://github.com/ziyezhou-Jerry/Project2-Stream-Compaction/blob/master/proj2_testing_output.png?raw=true) >>>

### Extra Credit
I have also Implemented the Radix Sort. Within the method, I am using the thrust::scan method to do the prefix-sum. The comparison with the CPU STL sort can be seen below:
![alt tag](https://github.com/ziyezhou-Jerry/Project2-Stream-Compaction/blob/master/proj2_extra_output.png?raw=true)

The input I used is manually generate by this for loop:
``` C++
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
```
The output is the sorted array, we can see it from the screenshot above.


