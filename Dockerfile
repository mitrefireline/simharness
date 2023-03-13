FROM continuumio/miniconda3:latest

ENV CODE_DIR=/code/

WORKDIR $CODE_DIR
COPY . .

RUN conda config --set ssl_verify false &&\
    conda env create -f conda-env.yaml &&\
    mkdir /root/.local &&\
    mkdir /root/.local/bin &&\
    apt update -y &&\
    apt install curl gcc -y

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "sh2", "/bin/bash", "-c"]
ENV PATH=$PATH:/root/.local/bin
ENV PATH=/opt/conda/envs/sh2/bin:$PATH

RUN curl -sSkL https://install.python-poetry.org | python - &&\
    sed -i '$d' /root/.bashrc &&\
    echo "conda activate sh2" >> /root/.bashrc &&\
    echo "conda activate sh2" >> /root/.profile &&\
    poetry install

CMD [ "echo", "SimHarness2 Container"]