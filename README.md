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
