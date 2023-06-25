####################################################################################################
# The core idea of this solution is to give a way to solve the following tasks:
# 1. To compute in CUDA any size and any count of dimensions array
# 2. To use batching to compute any size array, even if it s big that can't be fitted in GPU memory
####################################################################################################

import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import unittest


class TestCudaScript(unittest.TestCase):

    def test_case_2(self):
        # In this test case, we pass in parameters and validate the result.
        dimension_sizes = [3, 8, 4]
        max_block_x = 4
        max_grid_x = 4
        batch_size = 20
        result = main(dimension_sizes, max_block_x, max_grid_x, batch_size)
        self.assertEqual(result, True)


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
    print('Shape: ', shape)
    print('Dimensions count: ', len(shape))

    arr = arr.reshape(-1, order=reshape_order)
    result_array = np.zeros(arr.shape, dtype=np.bool_, order=reshape_order)
    total_elements = arr.size

    batch_start = 0
    while batch_start < total_elements:
        print('\nBatch start: ', batch_start)

        gpu_arr = gpuarray.to_gpu(arr[batch_start:batch_start+batch_size])
        print('GPU array size: ', gpu_arr.size)

        gpu_shape = gpuarray.to_gpu(np.array(shape, dtype=np.uint32))
        block = (int(max_block_x), 1, 1)
        print('Block: ', block)
        grid = (int(min(np.ceil(gpu_arr.size / max_block_x), max_grid_x)), 1, 1)
        print('Grid: ', grid)

        step = grid[0] * block[0]
        print('Step: ', step)

        with open('unravelcuda.cu', 'r') as f:
            kernel_code = f.read()
        ker = SourceModule(kernel_code)
        loop_unrolling = ker.get_function("loop_unrolling")

        loop_unrolling(
            gpu_arr,
            gpu_shape,
            np.uint64(gpu_arr.size),
            np.uint64(arr.shape),
            np.uint64(len(shape)),
            np.uint64(step),
            np.uint8(0 if reshape_order=='C' else 1),
            np.uint64(batch_start),
            block=block,
            grid=grid
        )

        result_array[batch_start:batch_start+gpu_arr.size] = gpu_arr.get()
        batch_start += batch_size

    result_array = result_array.reshape(shape, order=reshape_order)
    ctx.pop()
    return result_array


def main(dimension_sizes, max_block_x, max_grid_x, batch_size):

    drv.init()
    dev = drv.Device(0)
    ctx = dev.make_context()

    reshape_order = 'C' # C or F

    shape = [int(size) for size in dimension_sizes]

    arr = np.zeros(shape, dtype=np.bool_, order=reshape_order)

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

    # Print arr
    # print('Array test: ', arr_test)
    # Print arr_new
    # print('Array new: ', arr_new)

    # Check the result
    result_check = np.array_equal(arr_new, arr_test)
    print('\nResult check: ', result_check)
    return result_check


if __name__ == '__main__':
    unittest.main()
