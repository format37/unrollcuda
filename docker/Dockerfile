FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
WORKDIR /app/scripts

RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y \
    && apt-get install -y gcc-9 g++-9

# Set up symlinks for gcc and g++
RUN rm /usr/bin/gcc && rm /usr/bin/g++ \
    && ln -s /usr/bin/gcc-9 /usr/bin/gcc \
    && ln -s /usr/bin/g++-9 /usr/bin/g++

COPY requirements.txt /app/scripts/requirements.txt
RUN python3 -m pip install -r requirements.txt --no-cache-dir