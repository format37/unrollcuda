import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import time


def init_array(shape):
    # Random booleand array
    arr = np.random.randint(0, 2, shape, dtype=np.bool_)
    return arr


def main():

    device_id = 1
    drv.init()
    dev = drv.Device(device_id)
    ctx = dev.make_context()

    # Max block x
    max_block_x = dev.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_X)
    # Max grid x
    max_grid_x = dev.get_attribute(drv.device_attribute.MAX_GRID_DIM_X)

    # Redefine max block x
    max_block_x = 8
    # Redefine max grid x
    max_grid_x = 2

    # Define the zeros boolean array
    shape = 30
    arr = init_array(shape)

    # Send array to gpu
    gpu_arr = gpuarray.to_gpu(arr)
    print('GPU array size: ', gpu_arr.size)

    # Block and grid
    block = (int(max_block_x), 1, 1)
    print('Block: ', block)
    grid = (int(min(np.ceil(gpu_arr.size/max_block_x),max_grid_x)), 1, 1)
    print('Grid: ', grid)

    # Step
    step = grid[0] * block[0]
    print('Step: ', step)

    # Steps count
    steps_count = int(np.ceil(gpu_arr.size / step))
    print('Steps count: ', steps_count)

    # Read the kernel function from the file
    with open('loop_unrolling_1d.cu', 'r') as f:
        kernel_code = f.read()
    ker = SourceModule(kernel_code)
    loop_unrolling = ker.get_function("loop_unrolling")

    # Call the kernel
    loop_unrolling(
        gpu_arr,
        np.uint64(shape),
        np.uint64(step),
        block=block,
        grid=grid
    )
    # Downloading the array...
    arr_new = gpu_arr.get()
    # Clean up
    ctx.pop()

    # print('Result:\n', arr_new)
    # print('Result:\n', 'arr', 'arr_new')
    # for i in range(0, len(arr)):
    #     print(arr[i], arr_new[i])

    # Check the result
    print('Check result: ', np.array_equal(arr, ~arr_new))


if __name__ == '__main__':
    main()
