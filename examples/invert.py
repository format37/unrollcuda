import numpy as np
from unrollcuda import unrollcuda


def main():

    dimensions = [3, 4]
    reshape_order = 'C' # C or F
    shape = [int(size) for size in dimensions]
    # random boolean values
    arr = np.random.choice(a=[False, True], size=shape, p=[0.5, 0.5])

    with open('invert.cu', 'r') as f:
        kernel_code = f.read()

    uc = unrollcuda(kernel_code)
    arr_new = uc.inference(arr)

    # Prepare the test array
    arr_test = arr.copy()
    print('\nOriginal array:\n', arr_test)
    # Convert all False values to True and vice versa
    arr_test = np.logical_not(arr_test)
    print('\nTest array:\n', arr_test)

    # Check the result
    result_check = np.array_equal(arr_new, arr_test)
    print('\nResult check: ', result_check)


if __name__ == '__main__':
    main()
