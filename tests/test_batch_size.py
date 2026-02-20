"""Issue 3: batch_size=0 must not be permanently mutated after inference().

Before fix: FAILS — inference() sets self.batch_size = arr.size when batch_size is 0,
            permanently changing the instance attribute.
After fix:  PASSES — a local variable is used; self.batch_size stays at 0.
"""
import numpy as np
from conftest import requires_gpu, DOUBLE_KERNEL


@requires_gpu
def test_batch_size_zero_preserved():
    """batch_size=0 (auto) should remain 0 after inference()."""
    import unrollcuda as uc

    ker = uc.kernel(DOUBLE_KERNEL, batch_size=0)
    assert ker.batch_size == 0

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ker.inference(arr)

    assert ker.batch_size == 0, (
        f"batch_size was mutated from 0 to {ker.batch_size}"
    )


@requires_gpu
def test_explicit_batch_size_preserved():
    """An explicit batch_size should remain unchanged after inference()."""
    import unrollcuda as uc

    ker = uc.kernel(DOUBLE_KERNEL, batch_size=10)
    arr = np.ones(25, dtype=np.float32)
    ker.inference(arr)

    assert ker.batch_size == 10, (
        f"batch_size was mutated from 10 to {ker.batch_size}"
    )


@requires_gpu
def test_batch_size_zero_works_with_different_sizes():
    """With batch_size=0, inference should work correctly for arrays of different sizes."""
    import unrollcuda as uc

    ker = uc.kernel(DOUBLE_KERNEL, batch_size=0)

    arr_small = np.array([1.0, 2.0], dtype=np.float32)
    result_small = ker.inference(arr_small)
    np.testing.assert_array_almost_equal(result_small, arr_small * 2)

    arr_large = np.ones(100, dtype=np.float32)
    result_large = ker.inference(arr_large)
    np.testing.assert_array_almost_equal(result_large, arr_large * 2)
