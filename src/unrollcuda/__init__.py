"""
Unrollcuda is a Python library that provides an efficient mechanism for performing computations on large multi-dimensional arrays using CUDA. It leverages loop unrolling and batching techniques to handle array sizes larger than GPU memory. 

Classes:
    unrollcuda: The main class of the unrollcuda library.

Class `unrollcuda` Arguments:
    kernel_code (str, required): The CUDA kernel code to be run on the GPU.
    gpu_id (int, optional): The ID of the GPU to be used for computations. Defaults to 0.
    reshape_order (str, optional): Defines the order in which elements of the input array are read. It could be 'C' for row-major (C-style) or 'F' for column-major (Fortran-style). Defaults to 'C'.
    max_block_x (int, optional): Maximum size of block dimension X. If 0, the maximum size will be determined by the GPU device. Defaults to 0.
    max_grid_x (int, optional): Maximum size of grid dimension X. If 0, the maximum size will be determined by the GPU device. Defaults to 0.
    batch_size (int, optional): Size of the batch used when the array can't be fitted into GPU memory at once. If 0, the batch size will be equal to the array size. Defaults to 0.
    verbose (bool, optional): If True, verbose logging is enabled. Defaults to False.

Project Links:
    PyPI: https://pypi.org/project/unrollcuda
    GitHub: https://github.com/format37/unrollcuda
"""
from .unrollcuda import unrollcuda