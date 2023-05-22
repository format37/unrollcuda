# CUDA tips
GPU device is nice for parallel computing, bit each GPU device have its limitations in available memory and core count. Does it means limits for computation? Of course, ~~not~~ yes.
But we can extend the limits with some tricks.  
To overcome limits with the minimal performance disadvantages, we can utilize:  
| Issue | Approach |
| ----- | -------- |
| N-dimension representation | loop unrolling |
| Extending the core count limits | addressing |
| Extending the memory limits | batching |
  
Let's start from case, when the memory is enough but the count of array elements is bigger than available max_grid_x which is always 2147483647 elements.  
Following the CUDA datatypes  sheet:  
| Length (bytes) | NumPy type   | CUDA type           | Max Value                     |
| -------------- | ------------ | ------------------- | ----------------------------- |
| 1              | np.bool      | bool                | 1                             |
| 1              | np.int8      | signed char         | 2<sup>7</sup>-1 == 127        |
| 2              | np.int16     | short               | 2<sup>15</sup>-1 == 32767     |
| 4              | np.int32     | int                 | 2<sup>31</sup>-1 == 2147483647|
| 8              | np.int64     | long long           | 2<sup>63</sup>-1 == 9223372036854775807 |
| 1              | np.uint8     | unsigned char       | 2<sup>8</sup>-1 == 255        |
| 2              | np.uint16    | unsigned short      | 2<sup>16</sup>-1 == 65535     |
| 4              | np.uint32    | unsigned int        | 2<sup>32</sup>-1 == 4294967295|
| 8              | np.uint64    | unsigned long long  | 2<sup>64</sup>-1 == 18446744073709551615 |
| 4              | np.float32   | float               | n/a                           |
| 8              | np.float64   | double              | n/a                           |
| 2              | np.complex64 | cuComplex           | n/a                           |
| 4              | np.complex128| cuDoubleComplex     | n/a                           |
  
We have more than 2'147'483'647 in array which is equal to np.int32. Suppose that we have 7 times bigger array of bool values. It will be a 15032385529 elements. Looks like unsigned long long can count this pointer. Yeah we still have limit of 18'446'744'073'709'551'615 elements, using unsigned long long. Ok, first of all, it is 8'589'934'596 times bigger than 2'147'483'647. In addition the bool array of 18'446'744'073'709'551'615 elements will require an exabytes of memory. Currently I can store only 44'073'709'551'615 with my 64Gb RAM:
```
x = da.ones((44073709551615,), chunks=(1000000,), dtype=bool)
```
That is the limits we have in 2023.  
  
Let's simplify our bench. I will define my max_grid_x as 10. Currently we don't need to mind about memory, then suppose that memory is enough.  
To operate the 30-sized array with a 10-sized max_grid_x, we will utilize loop-unrolling approach.

### Threads and cores and registers
In your CUDA kernel, each thread has its own private local memory. When you declare a variable within a thread, such as unsigned long long x;, it's stored in this local memory. The size of this local memory does not impact the global memory of the GPU, but it does affect the register usage. Each CUDA core has a limited number of registers, and if your kernel uses too many registers, it can limit the number of concurrent threads that can be run on each Streaming Multiprocessor (SM), reducing overall performance.

In this case, an unsigned long long is indeed 8 bytes, which corresponds to 64 bits. This does not inherently lead to a huge amount of GPU memory usage unless you have a very high number of threads each using such a variable. Remember, this memory usage is per-thread, not per-core. Each core doesn't hold a separate copy of the variable for each thread it might execute, but rather each thread has its own private copy of the variable.

## CUDA information
## 1D sample
## 1D addressing
### Grid dim limit
### Memory limit
## 2D addressing
## 3D addressing
## nD addressing
## Performance report