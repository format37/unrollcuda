# FROM format37/cuda:11.3.0-devel-ubuntu20.04
# FROM nvidia/cuda:11.3.0-devel-ubuntu20.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
# FROM nvidia/cuda:10.1-devel-ubuntu18.04
WORKDIR /app/data/scripts

RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y
RUN apt-get install python3-pip -y

# COPY requirements.txt /app/data/scripts/requirements.txt
# RUN python3 -m pip install -r requirements.txt --no-cache-dir
# RUN python3 -m pip install dask
# RUN python3 -m pip install distributed
# RUN python3 -m pip install PyMCubes
# RUN python3 -m pip install scikit-image
# RUN python3 -m pip install numba
CMD ["python3", "loop_unrolling.py"]
# CMD ["nvprof -f -o ../prof.nvvp python3 /app/data/scripts/visual_hull.py"]
# CMD ["cuda-gdb --args python3 -m pycuda.debug /app/data/scripts/visual_hull.py"]