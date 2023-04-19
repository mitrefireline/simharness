ARG RAY_VERSION=2.3.0

# Deployment Stage
FROM rayproject/ray:${RAY_VERSION}-py39-gpu as deploy

# Set the correct environment variables
ENV CUDA_VISIBLE_DEVICES=0 \
    HF_CACHE_DIR=None
# Copy the needed code
WORKDIR /code/
COPY requirements.txt README.md LICENSE main.py ./
COPY conf/ conf/
COPY simharness2/ simharness2/

RUN sudo apt-get update \
    && sudo apt-get install -y gcc \
        cmake \
        # libgtk2.0-dev \
        zlib1g-dev \
        # libgl1-mesa-dev \
        unzip \
        # unrar \
    && $HOME/anaconda3/bin/pip --no-cache-dir install -U pip \
    # First, install requirements
    && $HOME/anaconda3/bin/pip --no-cache-dir install -U \
           -r requirements.txt \
    && sudo apt-get clean

# Make sure tfp is installed correctly and matches tf version.
# RUN python -c "import tensorflow_probability"
