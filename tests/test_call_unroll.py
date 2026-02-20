"""Issue 5: call_unroll should have a clean method signature (no redundant self_tmp).

Before fix: FAILS — call_unroll(self, self_tmp, **kwargs) has a confusing redundant
            parameter, and the monkey-patching pattern is fragile.
After fix:  PASSES — clean call_unroll(self, **kwargs) signature; overrides use
            types.MethodType for proper binding.
"""
import inspect
import importlib.util
import os
import numpy as np
from conftest import requires_gpu, MULTIPLY_KERNEL


def _load_kernel_class():
    """Load the kernel class directly from source (no pip install required)."""
    src = os.path.join(
        os.path.dirname(__file__), '..', 'src', 'unrollcuda', 'unrollcuda.py'
    )
    spec = importlib.util.spec_from_file_location('unrollcuda_src', src)
    mod = importlib.util.module_from_spec(spec)
    # Don't actually execute the module (it imports pycuda at top level).
    # Just parse the source to inspect the class signature.
    import ast
    with open(src) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'call_unroll':
            return [arg.arg for arg in node.args.args]
    raise RuntimeError("call_unroll not found in source")


def test_call_unroll_no_redundant_self_param():
    """call_unroll should not have a 'self_tmp' parameter."""
    params = _load_kernel_class()
    assert 'self_tmp' not in params, (
        f"call_unroll has redundant 'self_tmp' parameter: {params}"
    )


@requires_gpu
def test_default_call_unroll():
    """Default call_unroll should work with a standard single-array kernel."""
    import unrollcuda as uc

    NEGATE_KERNEL = """\
__global__ void unroll(
    int *arr,
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
        arr[idx_full] = -arr[idx_full];
        i += 1;
        idx_full = i * step + idx;
    }
}
"""
    ker = uc.kernel(NEGATE_KERNEL)
    arr = np.array([1, -2, 3, -4], dtype=np.int32)
    result = ker.inference(arr)
    np.testing.assert_array_equal(result, np.array([-1, 2, -3, 4], dtype=np.int32))


@requires_gpu
def test_custom_call_unroll_with_method_type():
    """Overriding call_unroll via types.MethodType should work for multi-array kernels."""
    import types
    from pycuda import gpuarray
    import unrollcuda as uc

    def custom_call_unroll(self, **kwargs):
        gpu_arr1 = kwargs['arr1'].reshape(-1, order=self.reshape_order)
        gpu_arr1 = gpu_arr1[
            self.batch_start:self.batch_start + self.gpu_arr.size
        ]
        gpu_arr1 = gpuarray.to_gpu(gpu_arr1)
        self.unroll(
            self.gpu_arr, gpu_arr1, self.gpu_shape,
            self.gpu_arr_size, self.arr_size, self.len_shape,
            self.step, self.reshape_order_gpu, self.batch_start_gpu,
            block=self.block, grid=self.grid,
        )

    ker = uc.kernel(MULTIPLY_KERNEL)
    ker.call_unroll = types.MethodType(custom_call_unroll, ker)

    arr0 = np.array([2, 3, 4, 5], dtype=np.uint32)
    arr1 = np.array([5, 6, 7, 8], dtype=np.uint32)
    result = ker.inference(arr0, arr1=arr1)

    np.testing.assert_array_equal(
        result, np.array([10, 18, 28, 40], dtype=np.uint32)
    )
