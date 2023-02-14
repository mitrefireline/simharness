FROM python:3.9.16

ENV CODE_DIR=/code/

WORKDIR $CODE_DIR
COPY ./* ./

# Install anaconda, create the environment, install poetry, install the Python packages
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
    && bash Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -b -p /root/miniconda3 \
    && rm Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
    && /root/miniconda3/bin/conda env create --file conda-env.yaml \
    && curl -sSL https://install.python-poetry.org | /root/miniconda3/envs/sh2/bin/python - \
    && /root/.local/bin/poetry install