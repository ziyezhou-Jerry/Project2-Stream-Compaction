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
![alt tag](https://github.com/ziyezhou-Jerry/Project2-Stream-Compaction/blob/master/proj2_bottleneck_analysis.png?raw=true)

* Paste the output of the test program into a triple-backtick block in your
  README.

  Apart from the given test cases, I have also added my own test case for the radix sort. I used the STL sort method to get the CPU version of sort result and compare it with the GPU radix sort.
  <<<![alt tag](https://github.com/ziyezhou-Jerry/Project2-Stream-Compaction/blob/master/proj2_testing_output.png?raw=true) >>>

