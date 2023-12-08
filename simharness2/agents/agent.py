import logging
from typing import Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s")
)
logger.addHandler(handler)
logger.propagate = False


@dataclass
class ReactiveAgent:
    """A simple agent that reacts to its environment.

    FIXME: update docstring style, using llama2 suggestion for now.
    Parameters
    ----------
    agent_id : int
        The unique ID of this agent.
    sim_id : int
        The unique ID of the simulation this agent belongs to.
    initial_position : tuple[int, int]
        The (x,y) starting position of the agent, where (0,0) is the top-left corner of
        the map and (max_x, max_y) is the bottom-right corner of the map.

    Properties
    ----------
    x : int
        The current X coordinate of the agent.
    y : int
        The current Y coordinate of the agent.
    row : int
        The current row number where the agent resides.
    col : int
        The current column number where the agent resides.
    latest_movement : str or None
        The last movement made by the agent, if applicable.
    latest_interaction : str or None
        The last interaction had by the agent, if applicable.
    mitigation_placed : bool
        Whether the agent has placed any mitigations recently.
    moved_off_map : bool
        Whether the agent has moved off the map recently.

    """

    # NOTE: `agent_speed` ommitted, only used within `_do_one_simulation_step`
    # Attrs that should be specified on initialization
    agent_id: Any  # ex: "agent_0", "dozer_0", "handcrew_0", "ff_0", etc.
    sim_id: int  # should be contained within sim.agents.keys()
    initial_position: Tuple[int, int]

    # Attributes with default values
    latest_movement: int = None
    latest_interaction: int = None
    mitigation_placed: bool = False
    moved_off_map: bool = False

    def __post_init__(self):
        self._current_position = self.initial_position
        self.x, self.y = self.initial_position
        self.row, self.col = self.y, self.x

    @property
    def current_position(self) -> Tuple[int, int]:
        return self._current_position

    @current_position.setter
    def current_position(self, value: Tuple[int, int]):
        self._current_position = value
        self.x, self.y = value
        self.row, self.col = self.y, self.x

    @property
    def x(self) -> int:
        return self._current_position[0]

    @x.setter
    def x(self, value: int):
        self._current_position = (value, self.y)

    @property
    def y(self) -> int:
        return self._current_position[1]

    @y.setter
    def y(self, value: int):
        self._current_position = (self.x, value)

    @property
    def row(self) -> int:
        return self._current_position[1]

    @row.setter
    def row(self, value: int):
        self._current_position = (self.x, value)

    @property
    def col(self) -> int:
        return self._current_position[0]

    @col.setter
    def col(self, value: int):
        self._current_position = (value, self.y)

    def reset(self):
        self.latest_movement = None
        self.latest_interaction = None
        self.mitigation_placed = False
        self.moved_off_map = False
        self.__post_init__()
        # self.current_position = self.initial_position
        # self.reward = 0

    # def move(self, env: np.ndarray, direction: int) -> bool:
    #     """Moves the agent in the given direction if possible."""
    #     current_x, current_y = self.current_position
    #     dx, dy = self.actions[direction]
    #     next_x, next_y = current_x + dx, current_y + dy

    #     if env[next_y][next_x] == "_":
    #         self.current_position = (next_x, next_y)
    #         return True
    #     else:
    #         return False
