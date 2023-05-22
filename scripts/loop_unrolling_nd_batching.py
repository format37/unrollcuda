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
        dev,
        ctx
    ):

    print('Shape: ', shape)
    print('Dimensions count: ', len(shape))

    gpu_shape = gpuarray.to_gpu(np.array(shape, dtype=np.uint32))

    block = (int(max_block_x), 1, 1)
    print('Block: ', block)

    # Number of iterations required
    num_iterations = int(np.ceil(len(arr) / float(batch_size)))
    print(gpu_shape.size, batch_size, 'Number of iterations: ', num_iterations)

    # Process each batch
    for i in range(num_iterations):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        if end_idx > len(arr):
            end_idx = len(arr)

        arr_batch = arr[start_idx:end_idx]

        gpu_arr = gpuarray.to_gpu(arr_batch)
        print('GPU array size: ', gpu_arr.size)

        grid = (int(min(np.ceil(gpu_arr.size/max_block_x),max_grid_x)), 1, 1)
        print('Grid: ', grid)

        step = grid[0] * block[0]
        print('Step: ', step)

        steps_count = int(np.ceil(gpu_arr.size / step))
        print('Steps count: ', steps_count)

        with open('loop_unrolling_nd_batching.cu', 'r') as f:
            kernel_code = f.read()
        ker = SourceModule(kernel_code)
        loop_unrolling = ker.get_function("loop_unrolling")

        loop_unrolling(
            gpu_arr, # arr
            gpu_shape, # shape
            np.uint64(gpu_arr.size), # shape_total
            np.uint64(len(shape)), # shape_count
            np.uint64(step), # step
            np.uint64(start_idx), # start_idx
            np.uint8(0 if reshape_order=='C' else 1), # unsigned char order
            block=block,
            grid=grid
        )

        arr[start_idx:end_idx] = gpu_arr.get()

    ctx.pop()

    return arr



def main():

    drv.init()
    device_id = 0
    dev = drv.Device(device_id)
    ctx = dev.make_context()

    reshape_order = 'C'
        
    shape = [3,7,4,5,8]
    arr = np.zeros(shape, dtype=np.bool_, order=reshape_order)
    batch_size = 15
    # max_block_x = int(int(drv.Device(device_id).get_attribute(drv.device_attribute.MAX_BLOCK_DIM_X)))
    max_block_x = 8
    # max_grid_x = int(int(drv.Device(device_id).get_attribute(drv.device_attribute.MAX_GRID_DIM_X)))
    max_grid_x = 2    

    arr_new = unroll_cuda(
        reshape_order,
        shape,
        arr,
        batch_size,
        max_block_x,
        max_grid_x,
        dev,
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
