__global__ void loop_unrolling(
    bool *arr,
    unsigned long long shape,
    unsigned long long step
)
{
    unsigned long long idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long long i = 0;
    
    while (i * step + idx < shape) {
        arr[i * step + idx] = !arr[i * step + idx];
        i += 1;
    }
}