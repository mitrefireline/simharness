FROM python:3.9.16

# Install anaconda, create the environment, install poetry, install the Python packages
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
    && bash Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -b \
    && conda env create --file conda-env.yaml \
    && conda activate sh2 \
    && curl -sSL https://install.python-poetry.org | python - \
    && echo "export PATH=$PATH:$HOME/.local/bin" >> $HOME/.bashrc \
    && source ~/.bashrc
    && conda activate sh2 \
    && poetry install