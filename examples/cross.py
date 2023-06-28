import numpy as np
import unrollcuda as uc


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
    print('Array shape: ', arr.shape)

    with open('cross.cu', 'r') as f:
        kernel_code = f.read()
    # Define the unrollcuda instance
    ker = uc.kernel(kernel_code, verbose=False)
    # Call inference
    arr_new = ker.inference(arr)

    # Prepare the test array
    arr_test = arr.copy()
    # Set all elements on axis to True
    arr_test = test(arr_test)

    # Check the result
    # print('arr_test: ', arr_test)
    # print('arr_new: ', arr_new)
    result_check = np.array_equal(arr_new, arr_test)
    print('\nResult check: ', result_check)


if __name__ == '__main__':
    main()
