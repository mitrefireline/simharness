from dataclasses import dataclass
from typing import Dict, List, Optional, Union

@dataclass
class Env_Config:
    movements: List[str]
    interactions: List[str]
    attributes: List[str]
    normalized_attributes: List[str]
    agent_speed: int

@dataclass
class Sim_Harness:
    name: str
    simulation: str
    config: Env_Config

@dataclass
class Model:
    conv_filters: List[Union[int, List[int]]]
    post_fcnet_hiddens: List[int]

@dataclass
class Training:
    model: Model
    
@dataclass
class Resources:
    num_gpus: int
    num_cpus_per_worker: int
    num_gpus_per_worker: int
    num_cpus_for_local_worker: int
    num_trainer_workers: int
    num_gpus_per_trainer_worker: int
    num_cpus_per_trainer_worker: int
    local_gpu_idx: int
    custom_resources_per_worker: Dict
    placement_strategy: str
    
@dataclass
class Environment:
    render_env: bool
    clip_rewards: Optional[bool]
    normalize_actions: bool
    disable_env_checking: bool
    is_atari: bool