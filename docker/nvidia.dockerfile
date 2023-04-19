FROM continuumio/miniconda3:latest as conda

WORKDIR /code/

COPY conda-env.yaml .

RUN conda config --set ssl_verify false &&\
    conda env create -f conda-env.yaml &&\
    mkdir /root/.local &&\
    mkdir /root/.local/bin &&\
    apt update -y &&\
    apt install curl gcc -y &&\
    curl -ksSL https://gitlab.mitre.org/mitre-scripts/mitre-pki/raw/master/os_scripts/install_certs.sh | sh

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "sh2", "/bin/bash", "-c"]
ENV PATH=$PATH:/root/.local/bin
ENV PATH=/opt/conda/envs/sh2/bin:$PATH

COPY poetry.lock pyproject.toml .
COPY simharness2/ simharness2/

RUN curl -sSkL https://install.python-poetry.org | python - &&\
    sed -i '$d' /root/.bashrc &&\
    echo "conda activate sh2" >> /root/.bashrc &&\
    echo "conda activate sh2" >> /root/.profile &&\
    poetry install --only runtime

# Deployment Stage
FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04 as deploy
COPY --from=conda /opt/conda/envs/sh2 /venv/
COPY --from=conda /etc/ssl/certs/ /etc/ssl/certs/

# Set the correct environment variables
ENV CUDA_VISIBLE_DEVICES=0 \
    HF_CACHE_DIR=None \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt

# Symlink the venv python to the default PATH
RUN ln -s /venv/bin/python /usr/local/bin/python

# Copy the needed code
WORKDIR /code
COPY pyproject.toml poetry.lock README.md LICENSE test.py .
COPY simharness2/ simharness2/

CMD ["python", "test.py"]
