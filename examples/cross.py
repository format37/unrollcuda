import numpy as np
from unrollcuda import unrollcuda


def test(arr):
    # Set all elements in the second position of each axis to True
    indices = [slice(None)] * arr.ndim
    for axis in range(arr.ndim):
        indices[axis] = 1  # 1 corresponds to the second position
        arr[tuple(indices)] = True
        indices[axis] = slice(None)  # reset to original state
    return arr


def main():

    dimensions = [3, 4]
    reshape_order = 'C' # C or F
    shape = [int(size) for size in dimensions]
    arr = np.zeros(shape, dtype=np.bool_, order=reshape_order)

    with open('cross.cu', 'r') as f:
        kernel_code = f.read()

    uc = unrollcuda(kernel_code)
    arr_new = uc.inference(arr)

    # Prepare the test array
    arr_test = arr.copy()
    # Set all elements on axis to True
    arr_test = test(arr_test)

    # Check the result
    result_check = np.array_equal(arr_new, arr_test)
    print('\nResult check: ', result_check)


if __name__ == '__main__':
    main()
