import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np


def set_second_to_true(arr):
    # Set all elements in the second position of each axis to True
    indices = [slice(None)] * arr.ndim
    for axis in range(arr.ndim):
        indices[axis] = 1  # 1 corresponds to the second position
        arr[tuple(indices)] = True
        indices[axis] = slice(None)  # reset to original state

    return arr


def unroll_cuda(
        reshape_order,
        shape,
        arr,
        batch_size,
        max_block_x,
        max_grid_x,
        ctx
    ):

    # Define the zeros boolean array
    print('Shape: ', shape)
    # Print diomensions count
    print('Dimensions count: ', len(shape))

    arr = arr.reshape(-1, order=reshape_order)

    result_array = np.zeros(arr.shape, dtype=np.bool_, order=reshape_order)

    # batch_step = int(np.ceil(arr.size / batch_size))
    # print('Batch step: ', batch_step)

    batch_start = 0

    while batch_start < arr.size:
        print('\nBatch start: ', batch_start)

        # Send array to gpu
        gpu_arr = gpuarray.to_gpu(arr[batch_start:batch_start+batch_size])
        print('GPU array size: ', gpu_arr.size)

        # Send shape to gpu
        gpu_shape = gpuarray.to_gpu(np.array(shape, dtype=np.uint32))

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
        with open('loop_unrolling_nd_batching.cu', 'r') as f:
            kernel_code = f.read()
        ker = SourceModule(kernel_code)
        loop_unrolling = ker.get_function("loop_unrolling")

        # Call the kernel
        loop_unrolling(
            gpu_arr, # arr
            gpu_shape, # shape
            np.uint64(arr.shape), # shape_total
            np.uint64(len(shape)), # shape_count
            np.uint64(step), # step
            np.uint8(0 if reshape_order=='C' else 1), # unsigned char order
            np.uint64(batch_start), # batch_start
            block=block,
            grid=grid
        )
        # Downloading the array...
        result_array[ batch_start:batch_start+batch_size ] = gpu_arr.get()

        batch_start += batch_size - 1

    # Reshape the array
    result_array = result_array.reshape(shape, order=reshape_order)

    # Clean up
    ctx.pop()

    return result_array


def main():

    drv.init()
    dev = drv.Device(0)
    ctx = dev.make_context()

    reshape_order = 'C' # C or F

    shape = [4,3]
    
    # Input dimensions count
    dimensions_count = int(input('Dimensions count: '))
    # Redefine shape
    shape = [int(input('Dimension size: ')) for i in range(dimensions_count)]    
    
    batch_size = 4*5

    arr = np.zeros(shape, dtype=np.bool_, order=reshape_order)
    # Max block x
    max_block_x = dev.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_X)
    # Max grid x
    max_grid_x = dev.get_attribute(drv.device_attribute.MAX_GRID_DIM_X)
    # Redefine max block x
    max_block_x = int(input('Max block x: '))
    # Redefine max grid x
    max_grid_x = int(input('Max grid x: '))
    # Redefine batch size
    batch_size = int(input('Batch size: '))
    # Redefine array size

    print('Max block x: ', max_block_x)
    print('Max grid x: ', max_grid_x)
    print('Batch size: ', batch_size)
    print('Array size: ', arr.size)

    arr_new = unroll_cuda(
        reshape_order,
        shape,
        arr,
        batch_size,
        max_block_x,
        max_grid_x,
        ctx
    )

    # Prepare the test array
    arr_test = arr.copy()
    # Set all elements on axis to True
    arr_test = set_second_to_true(arr_test)

    # Check the result
    print('\nResult check: ', np.array_equal(arr_new, arr_test))
    # Print arr
    # print('Array test: ', arr_test)
    # Print arr_new
    # print('Array new: ', arr_new)


if __name__ == '__main__':
    main()
