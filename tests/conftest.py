import pytest


def gpu_available():
    try:
        import pycuda.driver as drv
        drv.init()
        return drv.Device.count() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(
    not gpu_available(),
    reason="CUDA GPU not available",
)

# Simple kernel that doubles each element (float32)
DOUBLE_KERNEL = """\
__global__ void unroll(
    float *arr,
    unsigned int *shape,
    unsigned long long gpu_arr_size,
    unsigned long long shape_total,
    unsigned long long dimensions_count,
    unsigned long long step,
    unsigned char order,
    unsigned long long batch_start
)
{
    unsigned long long idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long idx_full;
    unsigned int i = 0;

    idx_full = i * step + idx;

    while (idx_full < shape_total && idx_full < gpu_arr_size)
    {
        arr[idx_full] = arr[idx_full] * 2.0f;
        i += 1;
        idx_full = i * step + idx;
    }
}
"""

# Kernel that multiplies two uint32 arrays elementwise
MULTIPLY_KERNEL = """\
__global__ void unroll(
    unsigned int *arr0,
    unsigned int *arr1,
    unsigned int *shape,
    unsigned long long gpu_arr_size,
    unsigned long long shape_total,
    unsigned long long dimensions_count,
    unsigned long long step,
    unsigned char order,
    unsigned long long batch_start
)
{
    unsigned long long idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long idx_full;
    unsigned int i = 0;

    idx_full = i * step + idx;

    while (idx_full < shape_total && idx_full < gpu_arr_size)
    {
        arr0[idx_full] = arr0[idx_full] * arr1[idx_full];
        i += 1;
        idx_full = i * step + idx;
    }
}
"""
