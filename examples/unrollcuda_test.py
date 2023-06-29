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


class kernel:
    def __init__(
            self, 
            kernel_code='',
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
        self.drv = drv
        self.drv.init()
        self.dev = self.drv.Device(self.gpu_id)
        self.ctx = self.dev.make_context()
        self.kernel_code = kernel_code
        self.reshape_order = reshape_order
        self.batch_size = batch_size
        if max_block_x == 0:
            self.max_block_x = self.dev.get_attribute(
                self.drv.device_attribute.MAX_BLOCK_DIM_X
                )
        if max_grid_x == 0:
            self.max_grid_x = self.dev.get_attribute(
                self.drv.device_attribute.MAX_GRID_DIM_X
                )
            
    def __del__(self):

        try:
            self.ctx.pop()
            self.ctx.detach()
        except:
            pass

    def log(self, msg):
        if self.verbose:
            self.logger.info(' '+msg)

    def call_unroll(
            self, 
            self_tmp,
            **kwargs
            ):
        self.unroll(
            self.gpu_arr, 
            self.gpu_arr, 
            self.gpu_shape,
            self.gpu_arr_size, 
            self.arr_size, 
            self.len_shape, 
            self.step, 
            self.reshape_order_gpu,            
            self.batch_start_gpu, 
            block=self.block,
            grid=self.grid
            )
    def inference(self, arr, **kwargs):
        """
        Perform computations on a provided array using the CUDA kernel specified in the `unrollcuda` instance.
        
        This method uses loop unrolling and batching techniques to efficiently handle array sizes larger than GPU memory. The computations are performed in batches, if needed, and the resulting array is reshaped back to the original shape before it is returned.
        
        Arguments:
            arr (numpy.ndarray, required): The input array on which the computations will be performed. This can be a multi-dimensional array of any size.
            
        Returns:
            numpy.ndarray: The resulting array after performing computations. The shape of this array will be the same as the input array `arr`.
        """
        if self.kernel_code == '':
            raise Exception('No kernel code provided')
        
        shape = arr.shape
        # self.result_array = np.zeros(arr.size, dtype=np.bool_, order=self.reshape_order)
        self.result_array = np.zeros(arr.size, dtype=arr.dtype, order=self.reshape_order)
        if self.batch_size == 0:
            self.batch_size = arr.size
        arr = arr.reshape(-1, order=self.reshape_order)
        total_elements = arr.size

        self.batch_start = 0
        while self.batch_start < total_elements:
            self.log('Batch start: '+str(self.batch_start))
            self.gpu_arr = gpuarray.to_gpu(arr[self.batch_start:self.batch_start+self.batch_size])
            self.gpu_shape = gpuarray.to_gpu(np.array(shape, dtype=np.uint32))
            self.block = (int(self.max_block_x), 1, 1)
            self.grid = (int(min(np.ceil(self.gpu_arr.size / self.max_block_x), self.max_grid_x)), 1, 1)
            self.step = self.grid[0] * self.block[0]
            kernel_source = SourceModule(self.kernel_code)
            self.unroll = kernel_source.get_function("unroll")
            self.gpu_arr_size = np.uint64(self.gpu_arr.size)
            self.arr_size = np.uint64(arr.size)
            self.len_shape = np.uint64(len(shape))
            self.step = np.uint64(self.step)
            self.reshape_order_gpu = np.uint8(0 if self.reshape_order=='C' else 1)
            self.batch_start_gpu = np.uint64(self.batch_start)            
            self.call_unroll(self, **kwargs)

            self.result_array[self.batch_start:self.batch_start+self.gpu_arr.size] = self.gpu_arr.get()
            self.batch_start += self.batch_size

        self.result_array = self.result_array.reshape(shape, order=self.reshape_order)
        self.ctx.pop()

        return self.result_array
