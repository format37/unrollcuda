__global__ void loop_unrolling(
    bool *arr,
    unsigned long long shape,
    unsigned char step
)
{
    unsigned long long idx = threadIdx.x + blockIdx.x * blockDim.x;
    // idx = threadIdx.x + blockIdx.x * blockDim.x
    unsigned char i = 0;
    while (threadIdx.x + blockIdx.x * blockDim.x < shape) {
        arr[threadIdx.x + blockIdx.x * blockDim.x] = !arr[threadIdx.x + blockIdx.x * blockDim.x];
        //idx += blockDim.x * gridDim.x;
        i += 1;
    }
}