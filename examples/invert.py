import numpy as np
import unrollcuda as uc


def main():
        
    dimensions = [3, 4, 7, 12]
    shape = [int(size) for size in dimensions]
    # random boolean values
    arr = np.random.choice(
        a=[False, True],
        size=shape,
        p=[0.5, 0.5],
        )
    print('Array shape: ', arr.shape)
    
    # Read the kernel code from the file
    with open('invert.cu', 'r') as f:
        kernel_code = f.read()
    # Define the unrollcuda instance
    ker = uc.kernel(kernel_code)
    # Call inference
    arr_new = ker.inference(arr)

    # Prepare the test array
    arr_test = arr.copy()
    # Convert all False values to True and vice versa
    arr_test = np.logical_not(arr_test)

    # Check the result
    result_check = np.array_equal(arr_new, arr_test)
    print('Data check: ', result_check)


if __name__ == '__main__':
    main()
