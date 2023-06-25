sudo docker run -it --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/scripts:/app/scripts \
    --gpus all \
    cuda-tips python3 /app/scripts/unravelcuda.py
# cuda-tips python3 /app/scripts/loop_unrolling_nd_batching.py