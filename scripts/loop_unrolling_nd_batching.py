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
        memory_batch_size,
        max_block_x,
        max_grid_x,
        ctx
    ):    

    # Define the zeros boolean array
    print('Shape: ', shape)
    # Print diomensions count
    print('Dimensions count: ', len(shape))

    # TODO: Reshape if needed

    # Send shape to gpu
    gpu_shape = gpuarray.to_gpu(np.array(shape, dtype=np.uint32))

    # Block and grid
    block = (int(max_block_x), 1, 1)
    print('Block: ', block)
    grid = (int(min(np.ceil(memory_batch_size/max_block_x),max_grid_x)), 1, 1) # Only processing memory_batch_size elements at a time
    print('Grid: ', grid)

    # Step
    step = grid[0] * block[0]
    # print('Step: ', step)

    # Steps count
    steps_count = int(np.ceil(arr.size / step))
    print('Steps count: ', steps_count)

    # Read the kernel function from the file
    with open('loop_unrolling_nd.cu', 'r') as f:
        kernel_code = f.read()
    ker = SourceModule(kernel_code)
    loop_unrolling = ker.get_function("loop_unrolling")

    for i in range(steps_count):
        print('Step: ', i)
        # Determine the size of this batch
        batch_size = min(step, arr.size - i * step)

        # Allocate a smaller array on the device and transfer a batch into it
        gpu_arr = gpuarray.to_gpu(arr[i * step : i * step + batch_size])

        # Call the kernel
        loop_unrolling(
            gpu_arr, # arr
            gpu_shape, # shape
            np.uint64(gpu_arr.size), # shape_total
            np.uint64(len(shape)), # shape_count
            np.uint64(step), # step
            np.uint8(0 if reshape_order=='C' else 1), # unsigned char order
            block=block,
            grid=grid
        )

        # Transfer the results back
        arr[i * step : i * step + batch_size] = gpu_arr.get()

    # Clean up
    ctx.pop()

    return arr


def main():

    drv.init()
    dev = drv.Device(0)
    ctx = dev.make_context()

    reshape_order = 'C'

    memory_batch_size = 70
    shape = [23]
    arr = np.zeros(shape, dtype=np.bool_, order=reshape_order)
    # Max block x
    max_block_x = dev.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_X)
    # Max grid x
    max_grid_x = dev.get_attribute(drv.device_attribute.MAX_GRID_DIM_X)

    # Redefine max block x
    max_block_x = 8
    # Redefine max grid x
    max_grid_x = 2

    arr_new = unroll_cuda(
        reshape_order,
        shape,
        arr,
        memory_batch_size,
        max_block_x,
        max_grid_x,
        ctx
    )

    # Prepare the test array
    arr_test = arr.copy()
    # Set all elements on axis to True
    arr_test = set_second_to_true(arr_test)

    # Check the result
    print('Check result: ', np.array_equal(arr_new, arr_test))


if __name__ == '__main__':
    main()
