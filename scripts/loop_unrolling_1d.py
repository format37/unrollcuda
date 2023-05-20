import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import time

# Start time
start_time = time.time()

reshape_order = 'F'

device_id = 0
drv.init()
dev = drv.Device(device_id)
ctx = dev.make_context()

# Max block x
max_block_x = dev.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_X)

# Redefine max block x and max grid x
max_grid_x = 10
shape = 30
# Define the empty boolean array
arr = np.zeros(shape, dtype=np.bool_)

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

# Read the kernal function from the file
with open('kernel.cu', 'r') as f:
    kernel_code = f.read()
ker = SourceModule(kernel_code)
loop_unrolling = ker.get_function("loop_unrolling")

# Call the kernel
print('Calling the kernel...')
loop_unrolling(
    gpu_arr, # bool *arr
    np.uint64(shape), # unsigned long long shape
    block=block,
    grid=grid
)
print('Downloading the array...')
arr_new = gpu_arr.get()
# Clean up
print('Cleaning up...')
ctx.pop()

print(arr_new)

# End time
end_time = time.time()
# Print time
print('Time: ', end_time - start_time)
