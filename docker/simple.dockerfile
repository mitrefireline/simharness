ARG RAY_VERSION=2.3.0

# Deployment Stage
FROM rayproject/ray:${RAY_VERSION}-py39-cu116 as deploy

# Copy the needed code
WORKDIR /code/
COPY requirements.txt README.md LICENSE main.py ./
COPY conf/ conf/
COPY simharness2/ simharness2/

RUN sudo apt-get update \
    && sudo apt-get install -y \
        build-essential \
        wget \
    && $HOME/anaconda3/bin/pip --no-cache-dir install -U pip \
    # Install requirements
    && $HOME/anaconda3/bin/pip --no-cache-dir install -U \
           -r requirements.txt \
    && sudo apt-get clean