import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Load your CUDA kernel
with open('mandelbrot.cu', 'r') as f:
    kernel_code = f.read()

mod = SourceModule(kernel_code)

# Get a function handle for the CUDA kernel
mandelbrot_kernel = mod.get_function("compute_mandelbrot")

# Prepare data
shape = 1000  # square size
scale = 0.01  # scaling factor
arr = np.zeros(shape * shape).astype(np.bool)  # Initialize the numpy array

# Allocate memory on the device
arr_gpu = cuda.mem_alloc(arr.nbytes)

# Copy the data to GPU
cuda.memcpy_htod(arr_gpu, arr)

# Launch the kernel (assuming a 1D grid of 256 blocks each containing 256 threads)
mandelbrot_kernel(arr_gpu, np.uint64(shape), np.float32(scale), block=(256, 1, 1), grid=(256, 1))

# Copy the result back to CPU
cuda.memcpy_dtoh(arr, arr_gpu)

# Reshape the array to create a 2D image
image = arr.reshape((shape, shape))

# At this point, you can use matplotlib or a similar library to display the image
