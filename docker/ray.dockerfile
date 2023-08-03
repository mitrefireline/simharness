FROM rayproject/ray:2.3.0-py39-gpu as deploy

ENV PATH=$PATH:$HOME/.local/bin \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0 \
    HF_CACHE_DIR=None

WORKDIR /code/

COPY poetry.lock pyproject.toml .

RUN sudo rm -rf /etc/apt/sources.list.d &&\
    sudo apt update &&\
    sudo apt install curl -y &&\
    sudo curl -ksSL https://install.python-poetry.org | python - &&\
    sudo $HOME/.local/bin/poetry install --only runtime

COPY README.md LICENSE test.py .
COPY simharness2/ simharness2/

CMD ["poetry", "run", "python", "test.py"]
