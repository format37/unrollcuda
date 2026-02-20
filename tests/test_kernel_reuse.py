"""Issue 2: kernel instances must be reusable across multiple inference() calls.

Before fix: FAILS — ctx.pop() at the end of inference() destroys the CUDA context,
            so a second inference() call raises a PyCUDA error.
After fix:  PASSES — context stays active for the kernel's lifetime.
"""
import numpy as np
from conftest import requires_gpu, DOUBLE_KERNEL


@requires_gpu
def test_inference_called_twice():
    """A kernel should support calling inference() more than once."""
    import unrollcuda as uc

    ker = uc.kernel(DOUBLE_KERNEL)

    arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result1 = ker.inference(arr1)
    np.testing.assert_array_almost_equal(result1, arr1 * 2)

    arr2 = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    result2 = ker.inference(arr2)
    np.testing.assert_array_almost_equal(result2, arr2 * 2)


@requires_gpu
def test_inference_multidimensional_reuse():
    """Reuse should also work with multi-dimensional arrays."""
    import unrollcuda as uc

    ker = uc.kernel(DOUBLE_KERNEL)

    arr1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result1 = ker.inference(arr1)
    np.testing.assert_array_almost_equal(result1, arr1 * 2)
    assert result1.shape == arr1.shape

    arr2 = np.ones((3, 3, 3), dtype=np.float32)
    result2 = ker.inference(arr2)
    np.testing.assert_array_almost_equal(result2, arr2 * 2)
    assert result2.shape == arr2.shape
