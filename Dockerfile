FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
WORKDIR /app/scripts

RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y
RUN apt-get install python3-pip -y

COPY requirements.txt /app/scripts/requirements.txt
RUN python3 -m pip install -r requirements.txt --no-cache-dir
# CMD ["python3", "sample.py"]
# CMD ["nvprof -f -o ../prof.nvvp python3 test.py"]
# CMD ["cuda-gdb --args python3 -m pycuda.debug test.py"]