# SimHarness-2: Modular Reinforcement Learning Harness for Natural Disaster Modelers (RLlib)

<figure>
    <p align="center">
        <p align="center">
            <img src="assets/icons/simharness_logo.png">
        </p>
</figure>

# Introduction

SimHarness is a modular reinforcement learning harness based on the RLlib framework written in PyTorch made to interact with natural disaster modelers.
SimHarness's easy-to-use interface allows for the quick and simple training of intelligent agents within any simulation that implements the required API interface, such as [SimFire](https://gitlab.mitre.org/fireline/simfire).

# Installation
Clone the repository

```shell
git clone https://gitlab.mitre.org/fireline/reinforcementlearning/simharness-2.git
cd simharness/
```

Create a conda environment

```shell
conda env create --file conda-env.yml
conda activate sh
```

Install poetry

```shell
curl -sSL https://install.python-poetry.org | python -
echo "export PATH=$PATH:$HOME/.local/bin" >> $HOME/.bashrc
source $HOME/.bashrc
conda activate sh
```

Install remaining requirements

```shell
poetry install --without dev,docs,coverage
```
