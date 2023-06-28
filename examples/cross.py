import numpy as np
import unrollcuda as uc
import mcubes
import trimesh


def save_obj(voxels):
    filename = 'cross.obj'
    # Invert voxels
    voxels = np.invert(voxels)
    # set all border voxels to 1
    voxels[0] = 1
    voxels[-1] = 1
    voxels[:, 0] = 1
    voxels[:, -1] = 1
    voxels[:, :, 0] = 1
    voxels[:, :, -1] = 1
    # Generate vertices and triangles using marching cubes algorithm
    vertices, triangles = mcubes.marching_cubes(voxels, 0.999)        
    mcubes.export_obj(vertices, triangles, filename)
    
    mesh = trimesh.load_mesh(filename)
    # Invert normals
    # mesh.vertex_normals = -mesh.vertex_normal
    # face_count = 100
    # mesh = mesh.simplify_quadric_decimation(face_count=face_count)
    mesh.export(filename)


def test(arr):
    # Set all elements in the second position of each axis to True
    indices = [slice(None)] * arr.ndim
    for axis in range(arr.ndim):
        indices[axis] = 3  # 3 corresponds to the second position
        arr[tuple(indices)] = True
        indices[axis] = slice(None)  # reset to original state
    return arr


def main():

    dimensions = [12, 9, 11]
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
    # Save the result
    save_obj(arr_new)


if __name__ == '__main__':
    main()
