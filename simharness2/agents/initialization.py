from typing import Dict, List, Tuple
from abc import ABC, abstractmethod

import numpy as np
from simharness2.agents.agent import ReactiveAgent


class AgentInitializer(ABC):
    @abstractmethod
    def initialize_agents(
        self,
        *,
        agent_ids: List[str],
        sim_ids: np.ndarray,
        fire_map_shape: Tuple[int, int],
        **kwargs,
    ) -> Dict[str, ReactiveAgent]:
        pass


class FixedPositionAgentInitializer(AgentInitializer):
    def __init__(self, agent_pos: List[List[int]], **kwargs):
        self._agent_pos = agent_pos

    def initialize_agents(
        self,
        *,
        agent_ids: List[str],
        sim_ids: np.ndarray,
        fire_map_shape: Tuple[int, int],
        **kwargs,
    ) -> Dict[str, ReactiveAgent]:
        agent_dict = {}
        for agent_str, pos, sim_id in zip(agent_ids, self._agent_pos, sim_ids):
            x, y = pos
            agent = ReactiveAgent(agent_str, sim_id, (x, y), fire_map_shape)
            agent_dict[agent_str] = agent
        return agent_dict


class RandomPositionAgentInitializer(AgentInitializer):
    def __init__(self, x_range: List[int], y_range: List[int], **kwargs):
        self._x_range = x_range
        self._y_range = y_range

    def initialize_agents(
        self,
        *,
        agent_ids: List[str],
        sim_ids: np.ndarray,
        fire_map_shape: Tuple[int, int],
        **kwargs,
    ) -> Dict[str, ReactiveAgent]:
        agent_dict = {}
        for agent_str, sim_id in zip(agent_ids, sim_ids):
            x = np.random.randint(self._x_range[0], self._x_range[1])
            y = np.random.randint(self._y_range[0], self._y_range[1])
            agent = ReactiveAgent(agent_str, sim_id, (x, y), fire_map_shape)
            agent_dict[agent_str] = agent
        return agent_dict


class RandomEdgeAgentInitializer(AgentInitializer):
    def __init__(self, edges: List[str], x_range: List[int], y_range: List[int], **kwargs):
        self._edges = edges
        self._x_range = x_range
        self._y_range = y_range

    def _get_position(self, edge):
        if edge.lower() == "left":
            x = self._x_range[0]
            y = np.random.randint(self._y_range[0], self._y_range[1])
        elif edge.lower() == "right":
            x = self._x_range[1] - 1
            y = np.random.randint(self._y_range[0], self._y_range[1])
        elif edge.lower() == "top":
            x = np.random.randint(self._x_range[0], self._x_range[1])
            y = self._y_range[0]
        elif edge.lower() == "bottom":
            x = np.random.randint(self._x_range[0], self._x_range[1])
            y = self._y_range[1]
        elif edge.lower() == "random":
            edge = np.random.choice(["left", "right", "top", "bottom"])
            x, y = self._get_position(edge)
        else:
            raise ValueError("Error: unrecognized initialization side")
        return (x, y)

    def initialize_agents(
        self,
        *,
        agent_ids: List[str],
        sim_ids: np.ndarray,
        fire_map_shape: Tuple[int, int],
        **kwargs,
    ) -> Dict[str, ReactiveAgent]:
        agent_dict = {}
        for agent_str, edge, sim_id in zip(agent_ids, self._edges, sim_ids):
            pos = self._get_position(edge)
            agent = ReactiveAgent(agent_str, sim_id, pos, fire_map_shape)
            agent_dict[agent_str] = agent
        return agent_dict
