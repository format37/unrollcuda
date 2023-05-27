__global__ void loop_unrolling(
    bool *arr,
    unsigned int *shape,
    unsigned long long shape_total,
    unsigned long long dimensions_count,
    unsigned long long step,
    unsigned char order
)
{
    unsigned long long idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long idx_full;
    unsigned int i = 0;
    unsigned int *indices = new unsigned int[dimensions_count]; // array to hold the computed indices
    unsigned long long tmp;
    
    idx_full = i * step + idx;
    while (idx_full < shape_total)
    {
        tmp = idx_full;
        // Compute the indices
        for (unsigned int j = 0; j < dimensions_count; ++j)
        {
            unsigned int dimension = (order == 0) ? dimensions_count - j - 1 : j;
            // Modulo by the dimension size
            indices[dimension] = tmp % shape[dimension];
            // Divide by the dimension size
            tmp /= shape[dimension];
        }
        // Set true if any index equals to 1
        for (unsigned int j = 0; j < dimensions_count; ++j)
        {
            // j is the dimension
            if (indices[j] == 1)
            {
                arr[idx_full] = true;
                break;
            }
            arr[idx_full] = false;
        }
        i += 1;
        idx_full = i * step + idx;
    }
    // Free the memory
    delete[] indices;
}