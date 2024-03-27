from typing import Dict, Callable, Optional
import numpy as np

AGENT_INITIALIZATION_METHODS: Dict[str, Callable] = {}

def register_agent_initialization(name: Optional[str] = None):
    def decorator(func):
        global AGENT_INITIALIZATION_METHODS
        if name is None:
            init_method_name = func.__name__
        else:
            init_method_name = name
        AGENT_INITIALIZATION_METHODS[init_method_name] = func
        return func
    return decorator


@register_agent_initialization(name="manual")
def fixed_initialization(num_agents, pos_list):
    return pos_list


@register_agent_initialization(name="random")
def random_initialization(num_agents, width, height, start_x=0, start_y=0):
    pos_list = []
    for i in range(num_agents):
        x = np.random.randint(width) + start_x
        y = np.random.randint(height) + start_y
        pos_list.append([x, y])
    return pos_list
