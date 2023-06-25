####################################################################################################
# The core idea of this solution is to give a way to solve the following tasks:
# 1. Use Loop unrolling to compute in CUDA any size and any count of dimensions array
# 2. Use Batching to compute any size array, even if it s big that can't be fitted in GPU memory
####################################################################################################

import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import logging


logging.basicConfig(level=logging.INFO)


class unrollcuda:
    def __init__(
            self, 
            kernel_code,
            gpu_id=0, 
            reshape_order='C',
            max_block_x=0, 
            max_grid_x=0,
            batch_size=0,
            verbose=False
            ):
        self.verbose = verbose
        if self.verbose:            
            self.logger = logging.getLogger(__name__)

        self.gpu_id = gpu_id        
        drv.init()
        self.dev = drv.Device(self.gpu_id)
        self.ctx = self.dev.make_context()
        self.kernel_code = kernel_code
        self.reshape_order = reshape_order
        self.batch_size = batch_size
        if max_block_x == 0:
            self.max_block_x = self.dev.get_attribute(
                drv.device_attribute.MAX_BLOCK_DIM_X
                )
        if max_grid_x == 0:
            self.max_grid_x = self.dev.get_attribute(
                drv.device_attribute.MAX_GRID_DIM_X
                )

    def log(self, msg):
        if self.verbose:
            self.logger.info(' '+msg)
        
    def inference(self, arr):
        shape = arr.shape
        self.result_array = np.zeros(arr.size, dtype=np.bool_, order=self.reshape_order)
        if self.batch_size == 0:
            self.batch_size = arr.size
        arr = arr.reshape(-1, order=self.reshape_order)
        total_elements = arr.size

        batch_start = 0
        while batch_start < total_elements:
            self.log('Batch start: '+str(batch_start))
            gpu_arr = gpuarray.to_gpu(arr[batch_start:batch_start+self.batch_size])
            gpu_shape = gpuarray.to_gpu(np.array(shape, dtype=np.uint32))
            block = (int(self.max_block_x), 1, 1)
            grid = (int(min(np.ceil(gpu_arr.size / self.max_block_x), self.max_grid_x)), 1, 1)
            step = grid[0] * block[0]
            ker = SourceModule(self.kernel_code)
            unroll = ker.get_function("unroll")
            unroll(
                gpu_arr,
                gpu_shape,
                np.uint64(gpu_arr.size),
                np.uint64(arr.size),
                np.uint64(len(shape)),
                np.uint64(step),
                np.uint8(0 if self.reshape_order=='C' else 1),
                np.uint64(batch_start),
                block=block,
                grid=grid
            )

            self.result_array[batch_start:batch_start+gpu_arr.size] = gpu_arr.get()
            batch_start += self.batch_size

        self.result_array = self.result_array.reshape(shape, order=self.reshape_order)
        self.ctx.pop()

        return self.result_array
