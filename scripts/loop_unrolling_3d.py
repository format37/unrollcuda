import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import pycuda
import mcubes
import time
from datetime import datetime
import math

# Start time
start_time = time.time()

reshape_order = 'F'

device_id = 0
drv.init()
dev = drv.Device(device_id)
ctx = dev.make_context()

# Shared memory per block
shared_memory_per_block = dev.get_attribute(drv.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
# Constant memory
constant_memory = dev.get_attribute(drv.device_attribute.TOTAL_CONSTANT_MEMORY)
# Max block x
max_block_x = dev.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_X)
# Max grid x
max_grid_x = dev.get_attribute(drv.device_attribute.MAX_GRID_DIM_X)

print('[ Max block x: ', max_block_x)
print('[ Max grid x: ', max_grid_x)
# Redefine max block x and max grid x
# max_block_x = 1024
max_grid_x = 1000
print('+ Max block x: ', max_block_x)
print('+ Max grid x: ', max_grid_x)
multiplier = 10
shape_0 = 20*multiplier
shape_1 = 10*multiplier
shape_2 = 30*multiplier
# Define the empty boolean array
arr = np.zeros(shape_0 * shape_1 * shape_2, dtype=np.bool_)
arr_comp = np.ones(shape_0 * shape_1 * shape_2, dtype=np.bool_)

# Send grid to gpu
gpu_arr = gpuarray.to_gpu(arr)
print('GPU array size: ', gpu_arr.size)

# Define the grid
block = (int(max_block_x), 1, 1)
print('Block: ', block)
grid = (int(min(np.ceil(gpu_arr.size/max_block_x),max_grid_x)), 1, 1)
print('Grid: ', grid)
step = grid[0] * block[0]
print('Step: ', step)

# Steps count
steps_count = int(np.ceil(gpu_arr.size / step))
print('Steps count: ', steps_count)

# Define the kernel function
kernel_code = """
/* ### Define the datatypes carefully ###
Length (bytes) NumPy type	CUDA type
1 np.int8	    signed char 2**7-1 == 127
2 np.int16	    short 2**15-1 == 32767
4 np.int32	    int 2**31-1 == 2147483647
8 np.int64	    long long 2**63-1 == 9223372036854775807
1 np.uint8	    unsigned char 2**8-1 == 255
2 np.uint16	    unsigned short 2**16-1 == 65535
4 np.uint32     unsigned int 2**32-1 == 4294967295
8 np.uint64     unsigned long long 2**64-1 == 18446744073709551615
4 np.float32	float 2**32-1 == 4294967295
8 np.float64    double 2**64-1 == 18446744073709551615
*/
__global__ void test(
    bool *arr,
    unsigned long long shape_0,
    unsigned long long shape_1,
    unsigned long long shape_2,
    unsigned long long multiplier,
    unsigned char order
)
{
    unsigned long long idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long x, y, z;
    while (idx < shape_0 * shape_1 * shape_2) {
        if (order == 0)
        {
            // Order C
            x = idx % shape_2;
            y = (idx / shape_2) % shape_1;
            z = (idx / (shape_1 * shape_2)) % shape_0;
        }
        else
        {
            // Order F
            x = idx % shape_0;
            y = (idx / shape_0) % shape_1;
            z = (idx / (shape_0 * shape_1)) % shape_2;
            // alter F
            //z = idx / (shape_0 * shape_1);
            //y = (idx / shape_0) % shape_1;
            //x = idx % shape_0;
        }
        // Check if the current index is within the specified range
        if (x >= 2*multiplier && x < 3*multiplier || 
            y >= 2*multiplier && y < 3*multiplier ||
            z >= 2*multiplier && z < 3*multiplier) {
            arr[idx] = true;
        }
        idx += blockDim.x * gridDim.x;
    }
}
"""
ker = SourceModule(kernel_code)
gpu_test = ker.get_function("test")

# Call the kernel
print('Calling the kernel...')
gpu_test(
    gpu_arr,
    np.uint64(shape_0),
    np.uint64(shape_1),
    np.uint64(shape_2),
    np.uint64(multiplier),
    np.uint8(0 if reshape_order=='C' else 1),
    block=block,
    grid=grid
)
print('Downloading the array...')
arr_new = gpu_arr.get()
# Clean up
print('Cleaning up...')
ctx.pop()
# print('Comparing the arrays...')
# ds = arr_new.sum()
# print(ds, ds==arr_new.shape[0])
# Save as numpy array
# print('Saving as numpy array...')
# np.save('../loop_unrolling.npy', arr_new)

# Reshape the array
arr_new = arr_new.reshape(shape_0, shape_1, shape_2, order=reshape_order)

# Set borders to zero
arr_new[0,:,:] = 0
arr_new[-1,:,:] = 0
arr_new[:,0,:] = 0
arr_new[:,-1,:] = 0
arr_new[:,:,0] = 0
arr_new[:,:,-1] = 0

# Print shape and sum
print('Shape: ', arr_new.shape)
print('Sum: ', arr_new.sum())
# Save as obj 3d object
print('Genrating 3d object...')
vertices, triangles = mcubes.marching_cubes(arr_new, 0)
# Print len vertices
print('Vertices: ', len(vertices))
# Print len triangles
print('Triangles: ', len(triangles))
# Save as obj
print('Saving as obj...')
mcubes.export_obj(vertices, triangles, '../loop_unrolling.obj')

# End time
end_time = time.time()
# Print time
print('Time: ', end_time - start_time)
