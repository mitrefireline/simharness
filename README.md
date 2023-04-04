# SimHarness2: Modular Reinforcement Learning Harness for Natural Disaster Modelers (RLlib)

<figure>
    <p align="center">
        <p align="center">
            <img src="assets/icons/simharness2_logo.png">
        </p>
</figure>


# Introduction

SimHarness is a modular reinforcement learning harness based on the RLlib framework written in PyTorch made to interact with natural disaster modelers.
SimHarness's easy-to-use interface allows for the quick and simple training of intelligent agents within any simulation that implements the required API interface, such as [SimFire](https://gitlab.mitre.org/fireline/simfire).

# Installation
Clone the repository

```shell
git clone https://gitlab.mitre.org/fireline/reinforcementlearning/simharness2.git
cd simharness2/
```

Create a conda environment

```shell
conda env create --file conda-env.yaml
conda activate sh2
```

Install poetry

```shell
curl -sSL https://install.python-poetry.org | python -
echo "export PATH=$PATH:$HOME/.local/bin" >> $HOME/.bashrc
source $HOME/.bashrc
conda activate sh2
```

Install remaining requirements

```shell
poetry install
```

# Building Docker Image(s)

There are different flavors of docker images and only one of them ([simple](#simple)) currently works. The order of this section is by order of how close I think each dockerfile is to producing a working image once built.

## Simple

**File**: [`docker/simple.dockerfile`](docker/simple.dockerfile)

The simplest docker image just has the [`rayproject/ray:2.3.0-py39-gpu`](https://hub.docker.com/r/rayproject/ray) image as the base image and uses the [`requirements.txt`](requirements.txt) file to install the dependencies during build time. This doesn't install poetry or is using multi-stage builds because that was causing major headaches due to the way `ray` builds their docker images. The more complicated docker images below don't currently work but are described anyway.

The problem with this is that the [`requirements.txt`](requirements.txt) file has to be updated along with poetry to ensure the builds work. Not ideal.

**To build**:

```shell
docker build -f docker/simple.dockerfile .
```

## Ray

**File**: [`docker/ray.dockerfile`](docker/ray.dockerfile)

This was an attempt at using poetry to install the dependencies in the [`rayproject/ray:2.3.0-py39-gpu`](https://hub.docker.com/r/rayproject/ray) docker image, using it as a single stage build. This didn't work because I couldn't get the image to use the correct python when either installing the dependencies through poetry or running the test script after build when doing `docker run ...`.

**To build**:

```shell
docker build -f docker/ray.dockerfile .
```

## Multi

**File**: [`docker/multi.dockerfile`](docker/multi.dockerfile)

This is the most-tested multi-stage build dockerfile, but still does not work. It uses `continuumio/miniconda3:latest` as the build stage and `rayproject/ray:2.3.0-py39-gpu` as the deploy stage. I couldn't get the conda environment built in the first stage (using poetry) to correctly copy over to the second stage so that it could be used. This is because of how the `ray` image is built.


**To build**:

```shell
docker build -f docker/multi.dockerfile .
```

## Nvidia

**File**: [`docker/nvidia.dockerfile`](docker/nvidia.dockerfile)

This image is essentially the same as the [multi](docker/multi.dockerfile) image, but it uses `nvidia/cuda:11.2.0-runtime-ubuntu20.04` as the deploy image to get around the difficulties of `multi`. This didn't work either.

**To build**:

```shell
docker build -f docker/nvidia.dockerfile .
```

## Code Server

**File**: [`docker/code-server.dockerfile`](docker/code-server.dockerfile)

This was just the beginnings of putting the necessary packages into a docker image that also had code-server installed. Not tested all that much.

**To build**:

```shell
docker build -f docker/code-server.dockerfile .
```
