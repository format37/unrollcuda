#include <thrust/complex.h>

#define N 1000 // max iterations

__global__ void compute_mandelbrot(bool *arr, unsigned long long shape, float scale) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Loop unrolling
    for (unsigned int i = 0; i < shape; i += 2) {
        // Calculating coordinates
        float real = (idx % shape - shape / 2.0) * scale;
        float imag = (idx / shape - shape / 2.0) * scale;
        thrust::complex<float> c(real, imag);
        
        thrust::complex<float> z(0, 0);
        bool isInSet = true;
        
        // Calculate z = z^2 + c iteratively
        for (int j = 0; j < N; j++) {
            z = z * z + c;
            if (abs(z) > 2.0) {
                isInSet = false;
                break;
            }
        }

        arr[idx] = isInSet;

        // Unrolled loop's step
        idx++;
        if (idx >= shape * shape) break;

        real = (idx % shape - shape / 2.0) * scale;
        imag = (idx / shape - shape / 2.0) * scale;
        c = thrust::complex<float>(real, imag);
        z = thrust::complex<float>(0, 0);
        isInSet = true;

        for (int j = 0; j < N; j++) {
            z = z * z + c;
            if (abs(z) > 2.0) {
                isInSet = false;
                break;
            }
        }

        arr[idx] = isInSet;
    }
}
