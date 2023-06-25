sudo docker run -it --rm \
    -v $(pwd)/../examples:/app/scripts \
    --gpus all \
    cuda-tips python3 /app/scripts/cross.py
# cuda-tips python3 /app/scripts/loop_unrolling_nd_batching.py