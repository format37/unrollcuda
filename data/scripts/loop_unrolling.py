import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import mcubes
import time
from datetime import datetime, timedelta

# Start time
start_time = datetime.now()

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
print('Shared memory per block: ', shared_memory_per_block)
print('Constant memory: ', constant_memory)

# Redefine max block x and max grid x
# max_block_x = 3
max_grid_x = 100000000
print('Max block x: ', max_block_x)
print('Max grid x: ', max_grid_x)

shape_0 = 1000
shape_1 = 1000
shape_2 = 1000
# steps = 2
# Define the empty boolean array
arr = np.zeros(shape_0 * shape_1 * shape_2, dtype=np.bool_)

# Send grid to gpu
gpu_arr = gpuarray.to_gpu(arr)
print('GPU array size: ', gpu_arr.size)

# Define the grid
block = (int(max_block_x), 1, 1)
print('Block: ', block)
# grid = (int(min(np.ceil(gpu_arr.size/max_block_x),max_grid_x)), 1, 1)
max_grid_size = int(max_grid_x/max_block_x)
grid = (int(min(np.ceil(gpu_arr.size/max_block_x),max_grid_size)), 1, 1)
print('Grid: ', grid)

step = grid[0] * block[0]
print('Step: ', step)

# Steps count
steps_count = int(np.ceil(gpu_arr.size / step))
print('Steps count: ', steps_count)
# Define the kernel function
kernel_code = """__global__ void test(
        bool *arr,
        unsigned char shape_0,
        unsigned char shape_1,
        unsigned char shape_2,
        unsigned char step
        )
    {
        unsigned int idx, x, y, z;
        for (int s = 0; s < shape_0 * shape_1 * shape_2; s += step) {
            // arr is 2d array of shape_0 x shape_1
            idx = threadIdx.x + blockIdx.x * blockDim.x + s;
            if (idx > shape_0 * shape_1 * shape_2) return;
            
            arr[idx] = true;
            /*
            // set arr to true if x is 2
            x = idx % shape_0;
            if (x == 2) arr[idx] = true;
            // set arr to true if y is 2
            y = (idx / shape_0) % shape_1;
            if (y == 2) arr[idx] = true;
            // set arr to true if z is 2
            z = (idx / shape_0 / shape_1) % shape_2;
            if (z == 2) arr[idx] = true;
            */
        }
    }
"""
ker = SourceModule(kernel_code)
gpu_test = ker.get_function("test")

# Call the kernel
print('Calling the kernel...')
gpu_test(
    gpu_arr,
    np.uint8(shape_0),
    np.uint8(shape_1),
    np.uint8(shape_2),
    np.uint64(step),
    block=block,
    grid=grid
)
print('Downloading the array...')
arr_new = gpu_arr.get()
# Clean up
print('Cleaning up...')
ctx.pop()

# Save as numpy array
print('Saving as numpy array...')
np.save('loop_unrolling_3d.npy', arr_new)
"""
# Reashape the array to sqrt 2d
print('Reshaping the array...')
arr_new = arr_new.reshape((shape_0, shape_1, shape_2), order='F')

# Set the border values to 0
arr_new[0,:,:] = 0
arr_new[-1,:,:] = 0
arr_new[:,0,:] = 0
arr_new[:,-1,:] = 0
arr_new[:,:,0] = 0
arr_new[:,:,-1] = 0

# Save as obj file
vertices, triangles = mcubes.marching_cubes(arr_new, 0)
mcubes.export_obj(vertices, triangles, "loop_unrolling_3d.obj")"""

# End time
end_time = datetime.now()

print('Done in ', end_time - start_time, ' seconds')
