sudo docker run -it --rm \
    -v $(pwd)/../examples:/app/scripts \
    --gpus all \
    unrollcuda python3 /app/scripts/cross.py