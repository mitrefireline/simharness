FROM python:3.9.16

# Install anaconda, create the environment, install poetry, install the Python packages
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
    && bash Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -b -p /root/miniconda3 \
    && /root/miniconda3/bin/conda env create --file conda-env.yaml \
    && curl -sSL https://install.python-poetry.org | /root/miniconda3/envs/sh2/python - \
    && echo "export PATH=$PATH:$HOME/.local/bin" >> $HOME/.bashrc \
    && source ~/.bashrc \
    && /root/miniconda3/envs/sh2/bin/python -m poetry install