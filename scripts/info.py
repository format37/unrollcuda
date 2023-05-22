import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import pycuda


def gpu_info(device_id, dev, ctx):
    # Device name
    name = dev.name()
    print('GPU Name: ', name)

    # Compute Capability
    compute_capability = dev.compute_capability()
    print('Compute Capability: ', compute_capability)
    
    # Number of multiprocessors
    num_multiprocessors = dev.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)
    print('Number of Multiprocessors: ', num_multiprocessors)
    
    # Threads per multiprocessor
    max_threads_per_sm = int(drv.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR)
    print('Max Threads Per Multiprocessor: ', max_threads_per_sm)
    
    # Threads per block
    threads_per_block = int(drv.device_attribute.MAX_THREADS_PER_BLOCK)
    print('Max Threads Per Block: ', threads_per_block)
    
    # Shared memory per block
    shared_memory_per_block = dev.get_attribute(drv.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
    print('Shared Memory Per Block: ', shared_memory_per_block)

    # Constant memory
    constant_memory = dev.get_attribute(drv.device_attribute.TOTAL_CONSTANT_MEMORY)
    print('Total Constant Memory: ', constant_memory)

    # Dimensions of blocks
    max_block_x = int(drv.Device(device_id).get_attribute(drv.device_attribute.MAX_BLOCK_DIM_X))
    print('Max Block Dim X: ', max_block_x)
    max_block_y = int(drv.Device(device_id).get_attribute(drv.device_attribute.MAX_BLOCK_DIM_Y))
    print('Max Block Dim Y: ', max_block_y)
    max_block_z = int(drv.Device(device_id).get_attribute(drv.device_attribute.MAX_BLOCK_DIM_Z))
    print('Max Block Dim Z: ', max_block_z)

    # Dimensions of grid
    max_grid_x = int(drv.Device(device_id).get_attribute(drv.device_attribute.MAX_GRID_DIM_X))
    print('Max Grid Dim X: ', max_grid_x)
    max_grid_y = int(drv.Device(device_id).get_attribute(drv.device_attribute.MAX_GRID_DIM_Y))
    print('Max Grid Dim Y: ', max_grid_y)
    max_grid_z = int(drv.Device(device_id).get_attribute(drv.device_attribute.MAX_GRID_DIM_Z))
    print('Max Grid Dim Z: ', max_grid_z)

    # Clock rate
    clock_rate = dev.get_attribute(drv.device_attribute.CLOCK_RATE)
    print('Clock Rate: ', clock_rate)

    # Memory information
    total_memory = dev.total_memory()
    print('Total Memory: ', total_memory)

    free_memory = drv.mem_get_info()[0]
    print('Free Memory: ', free_memory)


def main():
    device_id = 0
    drv.init()
    dev = drv.Device(device_id)
    ctx = dev.make_context()
    gpu_info(device_id, dev, ctx)
    # Free memory
    ctx.pop()


if __name__ == '__main__':
    main()
