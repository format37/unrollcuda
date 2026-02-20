"""Issue 1: CUDA kernels must not call delete[] on stack-allocated arrays.

Before fix: FAILS — .cu files contain 'delete[] indices' on stack-allocated arrays (UB).
After fix:  PASSES — the offending lines are removed.
"""
import os
import glob


def test_no_delete_on_stack_arrays():
    """Example .cu files should not contain delete[] on stack-allocated arrays."""
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    cu_files = glob.glob(os.path.join(examples_dir, '*.cu'))
    assert len(cu_files) > 0, "No .cu files found in examples/"

    for filepath in cu_files:
        with open(filepath) as f:
            content = f.read()
        filename = os.path.basename(filepath)
        assert 'delete[]' not in content, (
            f"{filename}: 'delete[]' called on stack-allocated array — "
            "this is undefined behavior"
        )
