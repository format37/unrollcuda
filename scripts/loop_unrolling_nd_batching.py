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
        device_id,
        shape,
        arr,
        memory_batching_size
    ):
    drv.init()
    dev = drv.Device(device_id)
    ctx = dev.make_context()

    # Max block x
    max_block_x = dev.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_X)
    # Max grid x
    max_grid_x = dev.get_attribute(drv.device_attribute.MAX_GRID_DIM_X)

    # Redefine max block x
    # max_block_x = 8
    # Redefine max grid x
    # max_grid_x = 2

    # Define the zeros boolean array
    print('Shape: ', shape)
    # Print diomensions count
    print('Dimensions count: ', len(shape))

    # Reshape to 1d
    arr_original = arr.reshape(-1, order=reshape_order)

    arr_new = np.empty(0, dtype=np.bool_)  # initialize outside the loop

    # Iterate each batch
    for batch in range(int(np.ceil(arr_original.size / memory_batching_size))):
        # Define arr as a part of the original array
        arr = arr_original[batch*memory_batching_size:(batch+1)*memory_batching_size]

        # Send array to gpu
        gpu_arr = gpuarray.to_gpu(arr)
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
            np.uint64(gpu_arr.size), # shape_total
            np.uint64(len(shape)), # shape_count
            np.uint64(step), # step
            np.uint8(0 if reshape_order=='C' else 1), # unsigned char order,
            np.uint64(batch), # batch
            np.uint64(memory_batching_size), # memory_batching_size
            block=block,
            grid=grid
        )
        # Downloading the array...
        arr_new = np.concatenate((arr_new, gpu_arr.get()))  # concatenate arrays
    # Clean up
    ctx.pop()

    # Reshape to original shape
    arr_new = arr_new.reshape(shape, order=reshape_order)

    return arr_new


def main():

    reshape_order = 'C'

    device_id = 0
    
    shape = [5,5]
    memory_batching_size = int(np.prod(shape)/2)
    print('Memory batching size: ', memory_batching_size)
    arr = np.zeros(shape, dtype=np.bool_, order=reshape_order)

    arr_new = unroll_cuda(
        reshape_order,
        device_id,
        shape,
        arr,
        memory_batching_size
    )

    # Prepare the test array
    arr_test = arr.copy()
    # Set all elements on axis to True
    arr_test = set_second_to_true(arr_test)

    # Check the result
    print('Check result: ', np.array_equal(arr_new, arr_test))
    print('Test: ', arr_test)
    print('Result: ', arr_new)
    


if __name__ == '__main__':
    main()
