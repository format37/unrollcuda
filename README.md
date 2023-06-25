# unrollcuda
Loop unrolling and batching for CUDA  
The core idea of this solution is to give a way to solve the following tasks:  
1. Use Loop unrolling to compute in CUDA any size and any count of dimensions array  
2. Use Batching to compute any size array, even if it s big that can't be fitted in GPU memory  