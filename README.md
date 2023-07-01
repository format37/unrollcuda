# unrollcuda
Loop unrolling and batching for CUDA  
The core idea of this solution is to give a way to solve the following tasks:  
1. Use Loop unrolling to compute in CUDA any size and any count of dimensions array.  
2. Use Batching to compute any size array, even if it s big that can't be fitted in GPU memory.  
## Disadvantages:
2. Batching leads to disability to access the all array in the kernel by the global index. Fortunately, batching is only need if array can't be fitted in GPU memory.
## Requirements:
[CUDA](https://developer.nvidia.com/cuda-downloads)  
[Python](https://www.python.org/downloads/)
## Getting Started
### Installation
```
pip install unrollcuda
```
### Usage
More examples are available at [github examples folder](https://github.com/format37/unrollcuda/tree/main/examples)
#### Invert values in a multi-dimensional boolean array
invert.cu
```
#define MAX_DIMENSIONS 4 // Set the number of dimensions accordingly to your array

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
    unsigned int indices[MAX_DIMENSIONS];
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
import numpy as np
import unrollcuda as uc


def main():
        
    dimensions = [2000, 100, 100, 100]
    shape = [int(size) for size in dimensions]
    # random boolean values
    arr = np.random.choice(
        a=[False, True],
        size=shape,
        p=[0.5, 0.5],
        )
    print('Array shape: ', arr.shape)
    print('Array size: ', arr.size)
    
    # Read the kernel code from the file
    with open('invert.cu', 'r') as f:
        kernel_code = f.read()
    # Define the unrollcuda instance
    ker = uc.kernel(kernel_code)
    # Call inference
    arr_new = ker.inference(arr)

    # Prepare the test array
    arr_test = arr.copy()
    # Convert all False values to True and vice versa
    arr_test = np.logical_not(arr_test)

    # Check the result
    result_check = np.array_equal(arr_new, arr_test)
    print('Data check: ', result_check)


if __name__ == '__main__':
    main()
```
#### Build a 3d cross object from 3d numpy array
cross.cu
```
#define MAX_DIMENSIONS 3 // Set the number of dimensions accordingly to your array

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
    unsigned int indices[MAX_DIMENSIONS];
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
        
        for (unsigned int j = 0; j < dimensions_count; ++j)
        {
            // j is the dimension

            // Your code there ++
            if (indices[j] == 3)
            {
                // Set true if any index equals to 3
                arr[idx_full] = true;
                break;
            }
            // Your code there --            
            
        }
        i += 1;
        idx_full = i * step + idx;
    }
    // Free the memory
    delete[] indices;
}
```
cross.py
```
import numpy as np
import unrollcuda as uc
import mcubes
import trimesh


def save_obj(voxels):
    filename = 'cross.obj'
    # Invert voxels
    voxels = np.invert(voxels)
    # set all border voxels to 1
    voxels[0] = 1
    voxels[-1] = 1
    voxels[:, 0] = 1
    voxels[:, -1] = 1
    voxels[:, :, 0] = 1
    voxels[:, :, -1] = 1
    # Generate vertices and triangles using marching cubes algorithm
    vertices, triangles = mcubes.marching_cubes(voxels, 0.999)        
    mcubes.export_obj(vertices, triangles, filename)
    
    mesh = trimesh.load_mesh(filename)
    # Invert normals
    # mesh.vertex_normals = -mesh.vertex_normal
    # face_count = 100
    # mesh = mesh.simplify_quadric_decimation(face_count=face_count)
    mesh.export(filename)


def test(arr):
    # Set all elements in the second position of each axis to True
    indices = [slice(None)] * arr.ndim
    for axis in range(arr.ndim):
        indices[axis] = 3  # 3 corresponds to the second position
        arr[tuple(indices)] = True
        indices[axis] = slice(None)  # reset to original state
    return arr


def main():

    dimensions = [12, 9, 11]
    reshape_order = 'C' # C or F
    shape = [int(size) for size in dimensions]
    arr = np.zeros(shape, dtype=np.bool_, order=reshape_order)
    print('Array shape: ', arr.shape)

    with open('cross.cu', 'r') as f:
        kernel_code = f.read()
    # Define the unrollcuda instance
    ker = uc.kernel(kernel_code, verbose=False)
    # Call inference
    arr_new = ker.inference(arr)

    # Prepare the test array
    arr_test = arr.copy()
    # Set all elements on axis to True
    arr_test = test(arr_test)

    # Check the result
    # print('arr_test: ', arr_test)
    # print('arr_new: ', arr_new)
    result_check = np.array_equal(arr_new, arr_test)
    print('\nResult check: ', result_check)
    # Save the result
    save_obj(arr_new)


if __name__ == '__main__':
    main()
```
#### Perfrorm an elementwise multiplication of two random int arrays
elementwise_mul.cu
```
#define MAX_DIMENSIONS 3 // Set the number of dimensions accordingly to your array

__global__ void unroll(
    unsigned int *arr0,
    unsigned int *arr1,
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
    unsigned long long i = 0;
    unsigned int indices[MAX_DIMENSIONS];
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
            // Multiply elementwise
            arr0[idx_full] = arr0[idx_full] * arr1[idx_full];
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
elementwise_mul.py
```
import numpy as np
import unrollcuda as uc
from pycuda import gpuarray
import time

def get_random_array(dimensions, start, end, dtype):
    # mean and standard deviation, based on start and end values
    mu, sigma = (start+end)/2, (end-start)/6
    shape = [int(size) for size in dimensions]
    arr = np.random.normal(mu, sigma, shape)
    arr = np.clip(arr, start, end)
    arr = arr.astype(dtype)
    return arr


def call_unroll(
        self, 
        **kwargs
        ):
    # Reshape the array to 1D
    gpu_arr1 = kwargs['arr1'].reshape(-1, order=self.reshape_order)
    # We need to split array the same way as the original array
    gpu_arr1 = gpu_arr1[
            self.batch_start:self.batch_start+self.batch_size
            ]
    # Send the array to GPU
    gpu_arr1 = gpuarray.to_gpu(gpu_arr1)
    self.log('self.block: '+str(self.block))
    self.log('self.grid: '+str(self.grid))
    # Call the kernel with the additiona array
    self.unroll(
        self.gpu_arr, 
        gpu_arr1, 
        self.gpu_shape,
        self.gpu_arr_size, 
        self.arr_size, 
        self.len_shape, 
        self.step, 
        self.reshape_order_gpu,
        self.batch_start_gpu, 
        block=self.block,
        grid=self.grid
        )
    

def main():
    start_time = time.time()
    dimensions = [2000, 1000, 1000]
    print('Generating arr0...')
    arr0 = get_random_array(dimensions, 0, 10, np.uint32)
    print('Generating arr1...')
    arr1 = get_random_array(dimensions, 0, 10, np.uint32)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Data preparation time: '+str(elapsed_time))
    
    # Read the kernel code from the file
    with open('elementwise_mul.cu', 'r') as f:
        kernel_code = f.read()
    
    # Define the unrollcuda instance
    ker = uc.kernel(kernel_code, verbose=True, batch_size=0) # Adjust batch_size to your GPU memory if it does not fit
    
    # Redefine the standard call_unroll method
    ker.call_unroll = call_unroll
    
    # Call inference with the new additional parameter
    start_time = time.time()
    arr_new = ker.inference(arr0, arr1=arr1)    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Cuda time: '+str(elapsed_time))
    
    # Check if the result is correct
    start_time = time.time()
    arr_new_check = np.multiply(arr0, arr1)
    # Check equality of the two arrays
    print(np.array_equal(arr_new, arr_new_check))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Numpy time: '+str(elapsed_time))


if __name__ == '__main__':
    main()
```