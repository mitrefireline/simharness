from dataclasses import dataclass
from typing import List, Union

@dataclass
class Environment:
    name: str
    sim_name: str
    movements: List[str]
    interactions: List[str]
    attributes: List[str]
    normalized_attributes: List[str]
    agent_speed: int
    
@dataclass
class Training:
    conv_filters: List[Union[int, List[int]]]
    post_fcnet_hiddens: List[int]