"""Issue 6: examples/unrollcuda_test.py should not be a copy of the library source.

Before fix: FAILS — the file is an exact duplicate of src/unrollcuda/unrollcuda.py.
After fix:  PASSES — the file is deleted.
"""
import os


def test_no_duplicate_source_in_examples():
    """examples/ should not contain a copy of the library source code."""
    test_file = os.path.join(
        os.path.dirname(__file__), '..', 'examples', 'unrollcuda_test.py'
    )
    source_file = os.path.join(
        os.path.dirname(__file__), '..', 'src', 'unrollcuda', 'unrollcuda.py'
    )

    if not os.path.exists(test_file):
        return  # File already removed — pass

    with open(test_file) as f:
        test_content = f.read()
    with open(source_file) as f:
        source_content = f.read()

    assert test_content != source_content, (
        "examples/unrollcuda_test.py is an exact copy of "
        "src/unrollcuda/unrollcuda.py — delete or replace it"
    )
