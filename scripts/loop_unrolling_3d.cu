__global__ void loop_unrolling(
    bool *arr,
    unsigned int *shape,
    unsigned long long shape_total,
    unsigned long long shape_count,
    unsigned long long step,
    unsigned char order
)
{
    unsigned long long idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long idx_full;
    unsigned int i = 0;
    unsigned int x, y, z;
    
    idx_full = i * step + idx;
    while (idx_full < shape_total)
    {
        if (order == 0)
        {
            // Order C
            x = idx_full % shape[2];
            y = (idx_full / shape[2]) % shape[1];
            z = ((idx_full / shape[2]) / shape[1]) % shape[0];
        }
        else
        {
            // Order F
            x = idx_full % shape[0];
            y = (idx_full / shape[0]) % shape[1];
            z = ((idx_full / shape[0]) / shape[1]) % shape[2];
        }
        // Set true for the second plane, the second row and the second column
        arr[idx_full] = x == 1 || y == 1 || z == 1;
        //printf("%u, %llu, %llu, %u, %u, %u\n",i, idx, idx_full, x, y, z);
        i += 1;
        idx_full = i * step + idx;
    }
}