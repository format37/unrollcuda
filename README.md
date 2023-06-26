# unrollcuda
Loop unrolling and batching for CUDA  
The core idea of this solution is to give a way to solve the following tasks:  
1. Use Loop unrolling to compute in CUDA any size and any count of dimensions array  
2. Use Batching to compute any size array, even if it s big that can't be fitted in GPU memory  
## Requirements:
[CUDA](https://developer.nvidia.com/cuda-downloads)  
[Python](https://www.python.org/downloads/)
## Getting Started
### Installation
```
pip install unrollcuda
```
### Usage
#### Invert values in a multi-dimensional boolean array
invert.cu
```
__global__ void unroll(
    bool *arr,
    unsigned int *shape,
    unsigned long long gpu_arr_size,
    unsigned long long shape_total,
    unsigned long long dimensions_count,
    unsigned long long step,
    unsigned char order,
    unsigned long long batch_start
)
{
    unsigned long long idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long idx_full;
    unsigned int i = 0;
    unsigned int *indices = new unsigned int[dimensions_count]; // array to hold the computed indices
    unsigned long long tmp;
    
    idx_full = i * step + idx;

    while (idx_full < shape_total && idx_full < gpu_arr_size)
    {
        tmp = idx_full + batch_start; // add batch_start to account for the offset
        // Compute the indices
        for (unsigned int j = 0; j < dimensions_count; ++j)
        {
            unsigned int dimension = (order == 0) ? dimensions_count - j - 1 : j;
            // Modulo by the dimension size
            indices[dimension] = tmp % shape[dimension];
            // Divide by the dimension size
            tmp /= shape[dimension];
        }
        //printf("idx_full: %llu, idx: %llu, batch_start: %llu\n", idx_full, idx, batch_start);
        
        for (unsigned int j = 0; j < dimensions_count; ++j)
        {
            // j is the dimension
            
            // Your code ++
            // Invert the value in arr
            arr[idx_full] = !arr[idx_full];
            // Your code --
            
            break;
        }
        i += 1;
        idx_full = i * step + idx;
    }
    // Free the memory
    delete[] indices;
}

```
invert.py
```
import unrollcuda as uc

```