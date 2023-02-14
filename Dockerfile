FROM python:3.9.16

ARG CI_PROJECT_DIR

WORKDIR $CI_PROJECT_DIR

# Install anaconda, create the environment, install poetry, install the Python packages
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
    && bash Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -b -p /root/miniconda3 \
    && rm Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
    && /root/miniconda3/bin/conda env create --file conda-env.yaml \
    && curl -sSL https://install.python-poetry.org | /root/miniconda3/envs/sh2/bin/python - \
    && echo "export PATH=$PATH:$HOME/.local/bin" >> $HOME/.bashrc \
    && /root/.local/bin/poetry install