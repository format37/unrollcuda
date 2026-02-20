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
        else:
            self.max_block_x = max_block_x
        if max_grid_x == 0:
            self.max_grid_x = self.dev.get_attribute(
                self.drv.device_attribute.MAX_GRID_DIM_X
                )
        else:
            self.max_grid_x = max_grid_x
            
    def __del__(self):

        try:
            self.ctx.pop()
            self.ctx.detach()
        except:
            pass

    def log(self, msg):
        if self.verbose:
            self.logger.info(' '+msg)

    def call_unroll(self, **kwargs):
        self.unroll(
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
The `inference` function in the `kernel` class executes computations on a provided numpy array using the CUDA kernel defined in the `kernel` class instance. The computations can handle multidimensional arrays of any size. If the array size is larger than what the GPU memory can accommodate, the function employs techniques of loop unrolling and batching to manage the computations efficiently. 

The function accepts a numpy array as an input and performs the specified computations in batches if necessary. After the computations, it reshapes the resulting array back to its original shape before returning it. 

The function signature is: `inference(self, arr, **kwargs)`

Parameters:

- `arr` (numpy.ndarray, required): This is the input array on which computations are performed. It can be a multidimensional array of any size.

- `**kwargs`: Any additional keyword arguments to be passed to the function.

Returns:

- `numpy.ndarray`: This is the output array after performing computations. Its shape is identical to the input array `arr`.

Please note, the function raises an exception if no kernel code is provided in the `kernel` class instance. Also, ensure the GPU is set up correctly and the CUDA drivers are installed, as the function does not handle GPU or CUDA related errors.

Usage example:

```python
import unrollcuda as uc

# Instantiate the kernel class
ker = uc.kernel(kernel_code)

# Create an array
arr = np.random.rand(1000, 1000)

# Use the inference function to perform computations on the array
arr_new = ker.inference(arr)
``` 

In this example, `kernel_code` would be the CUDA code you've written as a string, and `arr` is a 1000x1000 numpy array. The `inference` method performs computations on `arr` using the CUDA code provided and returns the resulting array.
        """
        if self.kernel_code == '':
            raise Exception('No kernel code provided')

        shape = arr.shape
        self.result_array = np.zeros(arr.size, dtype=arr.dtype, order=self.reshape_order)
        effective_batch_size = self.batch_size if self.batch_size > 0 else arr.size
        arr = arr.reshape(-1, order=self.reshape_order)
        total_elements = arr.size

        kernel_source = SourceModule(self.kernel_code)
        self.unroll = kernel_source.get_function("unroll")

        self.batch_start = 0
        while self.batch_start < total_elements:
            self.log('Batch start: '+str(self.batch_start))
            self.gpu_arr = gpuarray.to_gpu(arr[self.batch_start:self.batch_start+effective_batch_size])
            self.gpu_shape = gpuarray.to_gpu(np.array(shape, dtype=np.uint32))
            self.block = (int(self.max_block_x), 1, 1)
            self.grid = (int(min(np.ceil(self.gpu_arr.size / self.max_block_x), self.max_grid_x)), 1, 1)
            self.step = self.grid[0] * self.block[0]
            self.gpu_arr_size = np.uint64(self.gpu_arr.size)
            self.arr_size = np.uint64(arr.size)
            self.len_shape = np.uint64(len(shape))
            self.step = np.uint64(self.step)
            self.reshape_order_gpu = np.uint8(0 if self.reshape_order=='C' else 1)
            self.batch_start_gpu = np.uint64(self.batch_start)
            self.call_unroll(**kwargs)
            self.result_array[self.batch_start:self.batch_start+self.gpu_arr.size] = self.gpu_arr.get()
            self.batch_start += effective_batch_size

        self.result_array = self.result_array.reshape(shape, order=self.reshape_order)

        return self.result_array
