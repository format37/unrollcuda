# cuda_tips
GPU device is nice for parallel computing, bit each GPU device have its limitations in available memory and core count. Does it means limits for computation? Of course, not.  
To overcome limits with the minimal performance disadvantages, we can utilize adressing and batching.  
  
Let's start from case, when the memory is enough but the count of array elements is bigger than available max_grid_x which is always 2147483647 elements.  
Following the CUDA datatypes  sheet:  
| Length (bytes) | NumPy type   | CUDA type           | Max Value                     |
| -------------- | ------------ | ------------------- | ----------------------------- |
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
| 1              | np.bool      | bool                | n/a                           |
| 2              | np.complex64 | cuComplex           | n/a                           |
| 4              | np.complex128| cuDoubleComplex     | n/a                           |
  
We have more than 2147483647 in array which is equal to np.int32. Suppose that we have 7 times bigger array of bool values. It will be a 15032385529 elements. Looks like unsigned long long can count this pointer. Yeah we still have limit of 18446744073709551615 elements, using unsigned long long. Ok, first of all it is 8589934596 times bigger than 2147483647. And I going to manage this limit issue later.  
  
Let's simplify our bench. I will define my max_grid_x as 100.
To operate the 700-sized array, we will utilize loop-unrolling approach.

## CUDA information
## 1D sample
## 1D addressing
### Grid dim limit
### Memory limit
## 2D addressing
## 3D addressing
## nD addressing
## Performance report