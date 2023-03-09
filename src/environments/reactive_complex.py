# import random
from collections import OrderedDict as ordered_dict
from typing import Dict, List, OrderedDict, Tuple
from copy import deepcopy
import numpy as np
import torch
from simfire.sim.simulation import Simulation
from simfire.enums import BurnStatus
from .rl_harness import RLHarness
import random




class ReactiveComplexHarness(RLHarness):
    """

    This environment has a complex action space - 9
    This environment has a complex reward function utilizing burning, burned and mitigation spaces
    @Dhanuj

    Details
    ---------------
    - Agent Has 9 actions
    - Agent recieves high negative penalty for going into burning area
    - Agent recieves high negative penalty for placing mitigation in burned area
    - Agent recieves penalty for being close to burning fire
    - Agent is penalized at each timestep for the increased number of burning squares (includes mitigation as burning square)
    - Agent recieves positive reward for reducing overall number of burned squares from last run
    - Agent recieves positive reward for reducing burned squares in a faster number of steps
    

    This Environment Catcher will record all environments/actions
        for the RL return states.
    Inlcuding: Observation, Reward (or Penalty), "done", and any meta-data.

    This class will incorporate all gamelogic, input and rendering.

    It will also incorporate everything related to pygame into the render()
        method and separate init() and init_render() methods.

    We can then render ML routines using the step() and reset() method
        w/o loading the pygame package each step - if the environment is loaded,
        the rendering is not needed for training (saves execution time).

    Observation:
    ------------
    Type: gym.spaces.Dict(Box(low=min, high=max, shape=(255,255,len(max))))


    Num    Observation              min     max
    0      position                 0       1
    1      Fuel (w_0)               0       1
    2      Elevation                0       1 (float)
    3      mitigation               0       1



        In Reactive Case:
        -----------------
        Type: gym.spaces.Dict(Box(low=min, high=max, shape=(255,255,len(max))))
        Num    Observation              min     max
        0      position                 0       1
        1      Fuel (w_0)               0       1
        2      Burned/Unburned          0       1
        3      mitigation               0       3
        4      Burn Stats               0       6




    
    """

    def __init__(
        self,
        simulation: Simulation,
        actions: List[str],
        attributes: List[str],
        normalized_attributes: List[str],
        agent_speed: int,
    ) -> None:


        #Class Variables to be used in reward config

        self.reward_option = "7: Scaled Positive for Half Speed Fire" #"6: Scaled Positive"#"4: Positive" # "5: Burn Differential"
        self.expected_timesteps = 499
        ## Number of steps an agent has taken in the current game/simulation
        self.num_timesteps = 1
        self.num_agent_steps = 0
        self.agent_speed = 6 #agent_speed
        self.num_burned = 0
        self.lowest_burned = -1
        self.last_burning = 0
        self.last_burned = 0
        self.num_undamaged = 0
        self.num_mitigations = 0
        self.fastest_step_count = -1
        self.mitigation_placed = False
        self.agent_burning = False
        self.agent_in_burned_area = False

        #class variables not reset after each episode
        self.initial_episode = False
        self.episode_counter = 0
        super().__init__(simulation, actions, attributes, normalized_attributes)

    def step(self, action):
        if isinstance(action, (torch.Tensor, np.ndarray)):
            action = int(action.item())
        action_str = self.actions[action]
        reward = 0.0

        # If this action is an agent action, move the agent
        if action_str in self.nonsim_actions:
            pos_placeholder = self.agent_pos.copy()
            if action_str == "stay":
                pos_placeholder = pos_placeholder
                cut_fl = False
            elif action_str == "up-na":
                if not self.agent_pos[0] == 0:
                    pos_placeholder[0] -= 1
                cut_fl = False
            elif action_str == "down-na":
                if not self.agent_pos[0] == self.simulation.config.area.screen_size - 1:
                    pos_placeholder[0] += 1
                cut_fl = False
            elif action_str == "left-na":
                if not self.agent_pos[1] == 0:
                    pos_placeholder[1] -= 1
                cut_fl = False
            elif action_str == "right-na":
                if not self.agent_pos[1] == self.simulation.config.area.screen_size - 1:
                    pos_placeholder[1] += 1
                cut_fl = False
            elif action_str == "up-fl":
                if not self.agent_pos[0] == 0:
                    pos_placeholder[0] -= 1
                cut_fl = True
            elif action_str == "down-fl":
                if not self.agent_pos[0] == self.simulation.config.area.screen_size - 1:
                    pos_placeholder[0] += 1
                cut_fl = True
            elif action_str == "left-fl":
                if not self.agent_pos[1] == 0:
                    pos_placeholder[1] -= 1
                cut_fl = True
            elif action_str == "right-fl":
                if not self.agent_pos[1] == self.simulation.config.area.screen_size - 1:
                    pos_placeholder[1] += 1
                cut_fl = True
            else:
                None

            self.agent_pos = pos_placeholder
            if (
                self.state[self.attributes.index("fire_map")][self.agent_pos[0]][
                    self.agent_pos[1]
                ]
                == 0
            ) and cut_fl:
                self.simulation.update_mitigation(
                    [(self.agent_pos[1], self.agent_pos[0], 3)]
                )

            point = [self.agent_pos[1], self.agent_pos[0], 0]
            self.simulation.update_agent_positions([point])

        if self.num_agent_steps % self.agent_speed == 0:
            sim_fire_map, sim_active = self.simulation.run(1)
        else:
            sim_active = True
            sim_fire_map = self.simulation.fire_map

        fire_map = np.copy(sim_fire_map)

        #set the agent position
        if (fire_map[self.agent_pos[0]][self.agent_pos[1]]) == BurnStatus.BURNING:
            self.agent_burning = True
        if (fire_map[self.agent_pos[0]][self.agent_pos[1]]) == BurnStatus.BURNED:
            self.agent_in_burned_area = True

        fire_map[self.agent_pos[0]][self.agent_pos[1]] = len(self.actions) + 2 + 1
        self.state[self.attributes.index("fire_map")] = fire_map

        #update total number of burned squares
        #---------------------------------------------
        
        self.num_burned = np.count_nonzero(fire_map == BurnStatus.BURNED)
        self.num_undamaged = np.count_nonzero(fire_map == BurnStatus.UNBURNED)
        
        #---------------------------------------------
        


        #update total number of mitigations
        #---------------------------------------------
        if cut_fl:
            self.mitigation_placed = True
            self.num_mitigations = self.num_mitigations + 1

        
        
        #---------------------------------------------

        
        base_reward = self._calculate_reward(fire_map)
        
       
        

        #add negative reward if agent is in burning area or places mitigation in burned area
        
        #if self.agent_burning:
            #base_reward -= 1
        # if self.mitigation_placed:
        #     if self.agent_in_burned_area:
        #         base_reward -= 0.2

        reward += base_reward
       
        
       
        #give positive reward if fire ends faster in an episode than previous episodes
        if not sim_active:


            self.initial_episode = False

            self.episode_counter = self.episode_counter + 1
            # if self.episode_counter%25==0 or self.episode_counter < 10:
            #     print("Num Episodes: " + str(self.episode_counter))

            # print("Checking Enum Burned")
            # print(np.count_nonzero(fire_map == BurnStatus.BURNED))
            # print("Checking Enum Burning")
            # print(np.count_nonzero(fire_map == BurnStatus.BURNING))
            # print("Checking Enum Unburned")
            # print(np.count_nonzero(fire_map == BurnStatus.UNBURNED))
            # print("Checking Enum Fireline")
            # print(np.count_nonzero(fire_map == BurnStatus.FIRELINE))
            # print("Checking Enum Scratchline")
            # print(np.count_nonzero(fire_map == BurnStatus.SCRATCHLINE))
            # print("Checking Enum Wetline")
            # print(np.count_nonzero(fire_map == BurnStatus.WETLINE))
            # print("Checking Num Value Agent Position")
            # print(np.count_nonzero(fire_map == (len(self.actions) + 2 + 1)))
            # print("Checking Internal Mitigation Counter")
            # print(self.num_mitigations)
            # print("Checking Num Value Burned")
            # print(np.count_nonzero(fire_map == 2))
            # print("Checking Num Value Burning")
            # print(np.count_nonzero(fire_map == 1))
            # print("Checking Num Value Unburned")
            # print(np.count_nonzero(fire_map == 0))
            # print("Checking Num Value Fireline")
            # print(np.count_nonzero(fire_map == 3))
            # print("Checking Num Value Scratchline")
            # print(np.count_nonzero(fire_map == 4))
            # print("Checking Num Value Wetline")
            # print(np.count_nonzero(fire_map == 5))
            # print("Checking Raw Num Value Agent")
            # print(np.count_nonzero(fire_map == (12)))
            # print("Checking unique values in fire_map")
            # print(np.unique(fire_map))
            # print("Checking number of the unique values in fire_map")
            # print([(x, np.count_nonzero(fire_map == x)) for x in np.unique(fire_map)])
            # print("fire map size")
            # print(fire_map.size)
            # print("simulation fire map size")
            # print(self.simulation.fire_map.size)

            if self.reward_option == "6: Scaled Positive":
                if self.num_timesteps < self.expected_timesteps:
                    reward += (self.expected_timesteps - self.num_timesteps)
            if self.reward_option == "7: Scaled Positive for Half Speed Fire":
                if self.num_timesteps < self.expected_timesteps:
                    reward += (self.expected_timesteps - self.num_timesteps)

            #count mitigations as burned squares
            #self.num_burned = self.num_burned + self.num_mitigations
            self.num_burned = self.num_burned + np.count_nonzero(fire_map == BurnStatus.FIRELINE)

            if self.lowest_burned == -1:
                self.lowest_burned = self.num_burned

            if self.fastest_step_count == -1:
                self.fastest_step_count = self.num_agent_steps

            if self.num_burned < self.lowest_burned:
                
                #reward += 2
                #print(self.lowest_burned)
                #print(self.num_burned)
                self.lowest_burned = self.num_burned
                
                if self.num_agent_steps < self.fastest_step_count:
                    #reward += 2
                    self.fastest_step_count = self.num_agent_steps
        #    reward += 10

        if self.reward_option == "4: Positive" or self.reward_option == "6: Scaled Positive" or self.reward_option == "7: Scaled Positive for Half Speed Fire":
            if self._nearby_fire():
                 reward = 0
        elif self.reward_option == "5: Burn Differential":
            if self._nearby_fire():
                 reward = -2
             

        

        self.num_agent_steps += 1
        self.num_timesteps += 1

        #reset mitigation placed bool
        self.mitigation_placed = False
        self.agent_burning = False
        self.agent_in_burned_area = False
        return self.state, reward, not sim_active, {}

    def _nearby_fire(self) -> bool:
        nearby_locs = []
        screen_size = self.simulation.config.area.screen_size
        for i in range(self.agent_pos[0] - 1, self.agent_pos[0] + 2):
            for j in range(self.agent_pos[1] - 1, self.agent_pos[1] + 2):
                if (
                    i < 0
                    or i >= screen_size
                    or j < 0
                    or j >= screen_size
                    or [i, j] == self.agent_pos
                ):
                    pass
                else:
                    nearby_locs.append((i, j))
        for (i, j) in nearby_locs:
            if self.state[self.attributes.index("fire_map")][i][j] == 1:
                return True

        return False

    def _calculate_reward(self, fire_map: np.ndarray) -> float:


        
        #burning = np.count_nonzero(fire_map == 1)
        burning = np.count_nonzero(fire_map == BurnStatus.BURNING)
        burned = deepcopy(self.num_burned)

        #mitigations = np.count_nonzero(fire_map == BurnStatus.FIRELINE)
        #current_mitigations = np.count_nonzero(fire_map == 3)

        #calculate the differential to be the number of burning squares now vs the last timestep
        burning_differential = burning - self.last_burning 
        burned_differential = burned - self.last_burned
        self.last_burning = deepcopy(burning)
        self.last_burned = deepcopy(burned)

        #calculate the differential to be the number of mitigations now vs the last timestep
        #mitigation_differential = current_mitigations - self.num_mitigations
        #self.num_mitigations = deepcopy(current_mitigations)
        mitigation_differential = 0
        if self.mitigation_placed:
            mitigation_differential = 1

        
        

        total = self.simulation.config.area.screen_size**2
        

        if self.reward_option == "4: Positive":
            return ((self.num_undamaged * 1.0)/total)
        elif self.reward_option == "5: Burn Differential":
            return ((burning_differential + burned_differential + mitigation_differential) * -100.0/ total)
        elif self.reward_option == "6: Scaled Positive":
            # if self.num_timesteps == 1 or self.num_timesteps == 20 or self.num_timesteps == 40 or self.num_timesteps == 60 or self.num_timesteps == 80:
            #     print(self.num_timesteps)
            #     print(self.num_undamaged)
            #     print(((self.num_undamaged * 1.0)/total) * ((self.num_timesteps**1.7)/1000.0))
            timesteps = deepcopy(self.num_timesteps)
            if timesteps > 80:
                timesteps = 80 - (timesteps - 80)
            return (((self.num_undamaged * 1.0)/total) * ((timesteps**1.7)/10000.0))
        elif self.reward_option == "7: Scaled Positive for Half Speed Fire":
            if self.initial_episode:
                print(self.num_timesteps)
                print(self.num_undamaged)
                print(((self.num_undamaged * 1.0)/total) * ((self.num_timesteps**1.1)/1000.0))
            # if self.num_timesteps == 1 or self.num_timesteps == 20 or self.num_timesteps == 40 or self.num_timesteps == 60 or self.num_timesteps == 80:
            #      print(self.num_timesteps)
            #      print(self.num_undamaged)
            #      print(((self.num_undamaged * 1.0)/total) * ((self.num_timesteps**1.7)/1000.0))
            timesteps = deepcopy(self.num_timesteps)
            if timesteps > self.expected_timesteps:
                timesteps = self.expected_timesteps - (timesteps - self.expected_timesteps)
                if timesteps <= (self.expected_timesteps//2):
                    timesteps = (self.expected_timesteps//2)


            return (((self.num_undamaged * 1.0)/total) * ((((timesteps//self.agent_speed) + 1.0)**1.7)/(self.agent_speed * 10000.0)))
        else:
            return ((burning_differential + mitigation_differential) * -100.0/ total)
        
        

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        NOTE: reset() must be called before you can call step() for the first time.

        Terrain is received from the sim. Position matrix is assumed to be all 0's when
        received from sim. Updated to have agent at (0,0) on reset.

        Arguments:
            None

        Returns:
            `self.state`, a dictionary with the following structure:

        """
        self.num_burned = 0
        output = super().reset()
        point = [self.agent_pos[1], self.agent_pos[0], 0]
        self.simulation.update_agent_positions([point])

        #reset class variables
        self.num_agent_steps = 0
        self.num_timesteps = 1
        self.last_burning = 0
        self.num_undamaged = 0
        self.num_mitigations = 0
        self.last_burned = 0
        self.mitigation_placed = False
        self.agent_burning = False
        self.agent_in_burned_area = False
        
        
        return output

    def convert_sim_attributes(
        self, normalize_attributes: List[str]
    ) -> Tuple[OrderedDict[str, Tuple[int, int]], OrderedDict[str, np.ndarray]]:
        """
        This function will convert the returns of the Simulation.get_attributes()
            to the RL harness np.ndarray structure

        NOTE: 'elevation' attribute is scaled [-x,x] in config.yml but RL Harness
                expects 'elevation' on [0, 1] normalized scale

        Attributes:

            normalize_attributes: List[str]
                A list of strings of the desired attributes to normalize

        Returns:
            np.ndarray
                A numpy array of the converted attributes for the RL harness to use

        """
        return super().convert_sim_attributes(normalize_attributes)

    def hts_mitigation(self, mitigation_map: np.ndarray) -> np.ndarray:
        return super().hts_mitigation(mitigation_map)

    def sth_mitigation(self, mitigation_map: np.ndarray) -> np.ndarray:
        return super().sth_mitigation(mitigation_map)

    def get_nonsim_attribute_bounds(self) -> OrderedDict[str, Dict[str, int]]:
        nonsim_min_maxes = ordered_dict()
        # actions, unburned, burning, burned
        nonsim_min_maxes["fire_map"] = {"min": 0, "max": len(self.actions) + 2 + 1}

        return nonsim_min_maxes

    def get_nonsim_attribute_data(self) -> OrderedDict[str, np.ndarray]:
        nonsim_data = ordered_dict()

        nonsim_data["fire_map"] = np.zeros(
            (
                self.simulation.config.area.screen_size,
                self.simulation.config.area.screen_size,
            )
        )

        #set agent randomly in the upper left quadrant
        #random.seed(1234)
        self.agent_pos = [
             random.randrange(1, ((self.simulation.config.area.screen_size)//2)),
             random.randrange(1, ((self.simulation.config.area.screen_size)//2)),
        ]


        #self.agent_pos = [15, 15]


        nonsim_data["fire_map"][self.agent_pos[0]][self.agent_pos[1]] = (
            len(self.actions) + 2 + 1
        )
        self.simulation.update_mitigation([(self.agent_pos[1], self.agent_pos[0], 3)])

        return nonsim_data

    def render(self):
        self.simulation.rendering = True

    def action_masks(self) -> List[int]:
        mask = [1] * len(self.actions)
        screen_size = self.simulation.config.area.screen_size

        if self.agent_pos[0] - 1 < 0:
            mask[self.actions.index("up-na")] = 0
        if self.agent_pos[0] + 1 >= screen_size:
            mask[self.actions.index("down-na")] = 0
        if self.agent_pos[1] - 1 < 0:
            mask[self.actions.index("left-na")] = 0
        if self.agent_pos[1] + 1 >= screen_size:
            mask[self.actions.index("right-na")] = 0

        return mask

