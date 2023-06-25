####################################################################################################
# The core idea of this solution is to give a way to solve the following tasks:
# 1. To compute in CUDA any size and any count of dimensions array
# 2. To use batching to compute any size array, even if it s big that can't be fitted in GPU memory
####################################################################################################

import argparse
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import unittest
import random


class TestCudaScript(unittest.TestCase):
    
    def generate_random_tests(self, count, dimensions_count_max, dimension_value_max, max_block_x_max, max_grid_x_max, batch_size_max):
        random_tests = []
        for _ in range(count):
            dimension_sizes = [random.randint(2, dimension_value_max) for _ in range(random.randint(0, dimensions_count_max))]
            max_block_x = random.randint(4, max_block_x_max)
            max_grid_x = random.randint(4, max_grid_x_max)
            batch_size = random.randint(2, batch_size_max)
            random_tests.append((dimension_sizes, max_block_x, max_grid_x, batch_size))
        return random_tests

    def test_random_cases(self):
        random_tests = self.generate_random_tests(
            count=100,
            dimensions_count_max=6, 
            dimension_value_max=20, 
            max_block_x_max=300, 
            max_grid_x_max=300, 
            batch_size_max=1000
            )
        for dimension_sizes, max_block_x, max_grid_x, batch_size in random_tests:
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

    # Check the result
    result_check = np.array_equal(arr_new, arr_test)
    print('\nResult check: ', result_check)
    return result_check


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="run tests", action="store_true")
    return parser.parse_args()

def run_default():
    # set default parameters here
    default_dimension_sizes = [10, 10, 10]
    default_max_block_x = 256
    default_max_grid_x = 256
    default_batch_size = 512

    main(default_dimension_sizes, default_max_block_x, default_max_grid_x, default_batch_size)

if __name__ == '__main__':
    args = get_args()
    if args.test:
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    else:
        run_default()
