from typing import Dict, Tuple, Type

from simfire.sim.simulation import FireSimulation, Simulation
from simfire.utils.config import Config

_simulation_registry: Dict = {}


def register_simulation(
    name: str, simulation: Type[Simulation], train_config_path: str, eval_config_path: str
) -> None:
    """
    Register a new simulation class and associated config files into the simulation
    registry so it can be retrieved via string.

    Arguments:
        name (str): Name of the simulation class
        simulation (Type[Simulation]): Class implementation of the Simulation
        abstract class
        train_config_path (str): Path to the training configuration file for the
        simulation
        eval_config_path (str): Path to the evaluation configuration for the simulation

    Raises:
        ValueError: Assert that the simulation is an implementation of the
        Simulation class
        ValueError: Assert that the name for this simulation is not already taken
    """
    sub_class = None
    for cls in Simulation.__subclasses__():
        if issubclass(simulation, cls):
            sub_class = cls
            break
    if sub_class is None:
        raise ValueError(
            f"Error: the simulation {simulation} is not of any known subclasses of "
            "Simulation!"
        )

    if name in _simulation_registry:
        if _simulation_registry[name] != simulation:
            raise ValueError(
                f"Error: the name {name} is already registered for a different "
                "simulation, will not override."
            )

    train_config = Config(train_config_path)
    eval_config = Config(eval_config_path)
    _simulation_registry[name] = (simulation, train_config, eval_config)


def get_simulation_from_name(name: str) -> Tuple[Type[Simulation], Config, Config]:
    """
    Return the simulation class and config files associated with the given name.

    Arguments:
        name (str): Name of the requested simulation

    Raises:
        KeyError: Assert that the simulation has been registered with the
        simulation registry

    Returns:
        Tuple(Type[Simulation], Config, Config): Tuple of the simulation class and
        train/eval configs associated with the given name
    """
    if name not in _simulation_registry:
        raise KeyError(
            f"Error: unknown simulation type {name}, "
            "the only registed simulation types are: "
            f"{list(_simulation_registry.keys())}!"
        )
    return _simulation_registry[name]


register_simulation(
    "Fire-v0",
    FireSimulation,
    "./conf/simulation/simfire/train.yml",
    "./conf/simulation/simfire/eval.yml",
)
