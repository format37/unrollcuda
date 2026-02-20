"""Issue 4: CUDA kernel should be compiled once per inference(), not once per batch.

Before fix: FAILS — SourceModule is called inside the batch loop, so with 4 batches
            it is compiled 4 times instead of 1.
After fix:  PASSES — SourceModule is called once before the loop.
"""
import numpy as np
from unittest.mock import patch
from conftest import requires_gpu, DOUBLE_KERNEL


@requires_gpu
def test_kernel_compiled_once_with_multiple_batches():
    """SourceModule should be called exactly once regardless of batch count."""
    from pycuda.compiler import SourceModule
    import unrollcuda as uc

    ker = uc.kernel(DOUBLE_KERNEL, batch_size=5)
    arr = np.ones(20, dtype=np.float32)  # 20 / 5 = 4 batches

    with patch(
        'unrollcuda.unrollcuda.SourceModule', wraps=SourceModule
    ) as mock_sm:
        ker.inference(arr)

    assert mock_sm.call_count == 1, (
        f"SourceModule called {mock_sm.call_count} times, expected 1 "
        f"(4 batches should not trigger 4 compilations)"
    )


@requires_gpu
def test_single_batch_compiles_once():
    """Sanity check: a single-batch inference also compiles exactly once."""
    from pycuda.compiler import SourceModule
    import unrollcuda as uc

    ker = uc.kernel(DOUBLE_KERNEL, batch_size=0)
    arr = np.ones(10, dtype=np.float32)

    with patch(
        'unrollcuda.unrollcuda.SourceModule', wraps=SourceModule
    ) as mock_sm:
        ker.inference(arr)

    assert mock_sm.call_count == 1
