import logging
from typing import Optional, Tuple
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
    # NOTE: `agent_speed` ommitted, only used within `_do_one_simulation_step`
    # Attrs that should be specified on initialization
    agent_id: str  # ex: "agent_0", "dozer_0", "handcrew_0", "ff_0", etc.
    sim_id: int  # should be contained within sim.agents.keys()
    initial_position: Tuple[int, int]

    # Attributes with default values
    latest_movement: Optional[int] = None
    latest_interaction: Optional[int] = None
    mitigation_placed: bool = False
    moved_off_map: bool = False

    def __post_init__(self):
        self.current_position = self.initial_position
        # x,y pos, where (0,0) is top-left corner and (max_x, max_y) is bottom-right
        self.x, self.y = self.current_position
        self.row, self.col = self.y, self.x

        # Store the movement and interaction for the current timestep
        self.latest_movement: int = None
        self.latest_interaction: int = None
        # If the agent places a mitigation, this is set to True.
        self.mitigation_placed: bool = False
        # If the agent attempts to move out of bounds, this is set to True.
        self.moved_off_map: bool = False

        # actions: np.ndarray
        # reward: float = 0

    def reset(self):
        self.current_position = self.initial_position
        self.reward = 0

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
