FROM python:3.9.16

ARG CI_PROJECT_DIR
ENV CODE_DIR=/code/

RUN echo $CI_PROJECT_DIR
COPY --from=build "$CI_PROJECT_DIR/*" $CODE_DIR
WORKDIR $CODE_DIR

# Install anaconda, create the environment, install poetry, install the Python packages
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
    && bash Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -b -p /root/miniconda3 \
    && rm Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
    && /root/miniconda3/bin/conda env create --file conda-env.yaml \
    && curl -sSL https://install.python-poetry.org | /root/miniconda3/envs/sh2/bin/python - \
    && /root/.local/bin/poetry install