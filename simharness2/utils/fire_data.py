"""FIXME: A one line summary of the module or program.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
import logging
import os
import sys
import time
from typing import List, Tuple
import yaml
import modin.pandas as pd
import ray
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from simfire.enums import BurnStatus
from simfire.sim.simulation import FireSimulation

from simharness2.utils.simfire import get_simulator_hash


OmegaConf.register_new_resolver("operational_screen_size", lambda x: int(x * 39))
OmegaConf.register_new_resolver("calculate_half", lambda x: int(x / 2))
OmegaConf.register_new_resolver("square", lambda x: x**2)

logger = logging.getLogger(__name__)


# def get_fire_initial_position_data
def generate_fire_initial_position_data(
    sim: FireSimulation,
    save_path: str,
    output_size: int = 1,
    make_all_positions: bool = False,
):
    """Generate fire initial position data for a given fire simulation.

    Args:
        sim (FireSimulation): A FireSimulation instance.
        save_path (str): The path to save the generated data.
        output_size (int, optional): The number of fire initial positions to generate.
            Defaults to 1.
        make_all_positions (bool, optional): Whether to generate all possible fire
            initial positions. Defaults to False.
    """
    sim_hash_data = get_simulator_hash(sim.config, return_params_subset=True)
    sim_hash_value = sim_hash_data["hash_value"]
    sim_params_subset = sim_hash_data["params_subset"]
    save_path = os.path.join(save_path, sim_hash_value)
    logger.info(f"Data will be saved to: {save_path}")
    os.makedirs(save_path, exist_ok=True)

    coords = []
    # Create empty dataframe
    pos_df = pd.DataFrame({})
    # TODO: Move this if block into `load_fire_initial_position_data` (or similar).
    # Check if the fire initial position data has already been generated.
    if os.path.exists(os.path.join(save_path, "fire_initial_positions.csv")):
        logger.info(
            f"Fire initial position data for {sim_hash_value} has already been generated."
        )
        pos_df = pd.read_csv(os.path.join(save_path, "fire_initial_positions.csv"))
        total_pos = sim.fire_map.size if make_all_positions else output_size
        missing_pos = total_pos - len(pos_df)
        # Ensure all positions have been generated, and if not, generate them.
        if missing_pos > 0:
            logger.warning(
                f"Expected {output_size} positions, but only found {len(coords)}."
            )
            curr_pos = [tuple(row) for row in pos_df[["pos_x", "pos_y"]].values]
            # NOTE: excluding the current positions, so we must update output_size!
            coords = _generate_initial_positions(
                sim,
                output_size - len(curr_pos),
                make_all_positions,
                positions_to_exclude=curr_pos,
            )
        # No remaining calculations needed, just return the data
        else:
            logger.info(f"Found {len(pos_df)} positions, no more calculations needed.")
            return
            # TODO: What should we return? We can put the object in shared object store
            # and return object id. Or, we can return the data directly. Any others?
            # return pos_df

    # TODO: Move this if block into `generate_fire_initial_position_data` (or similar).
    else:
        # Write simulation config to file in save_path to aid interpretation of data.
        # FIXME: This is not the best place to do this.
        with open(os.path.join(save_path, "original_config.yaml"), "w") as f:
            logger.debug(f"Writing original config to {f.name}...")
            yaml.dump(sim.config.yaml_data, f)
        with open(os.path.join(save_path, "subset_config.yaml"), "w") as f:
            logger.debug(f"Writing params_subset to {f.name}...")
            yaml.dump(sim_params_subset, f)
        coords = _generate_initial_positions(sim, output_size, make_all_positions)

    if coords:
        # Ensure fire initial position type is static.
        sim.config.yaml_data["fire"]["fire_initial_position"]["type"] = "static"

        # Iterate over fire initial positions and run simfire (in parallel).
        results_refs = []
        logger.info(f"Running simfire for {len(coords)} fire initial positions...")
        for pos in coords:
            results_refs.append(_simulate_fire_propagation.remote(sim, pos))

        # Get results from Ray.
        parallel_returns = ray.get(results_refs)

        # Convert results to DataFrame and add new column, `percent_area_burned`
        df = pd.DataFrame(parallel_returns)
        df["percent_area_burned"] = df["burned"] / sim.fire_map.size
        # Check if we need to merge with the existing data.
        if not pos_df.empty:
            df = pd.concat([pos_df, df], ignore_index=True)
        # Save fire initial position data to CSV file.
        df.to_csv(os.path.join(save_path, "fire_initial_positions.csv"), index=False)


def _generate_initial_positions(
    sim: FireSimulation,
    output_size: int,
    make_all_positions: bool,
    positions_to_exclude: List[Tuple[int, int]] = [],
) -> List[Tuple[int, int]]:
    """Generate fire initial position candidates to run the simulation on."""
    logger.info("Generating the fire initial position candidates...")
    coords = []
    if make_all_positions:
        logger.info("The `output_size` will be ignored, as `make_all_positions` is True.")
        num_rows, num_cols = sim.fire_map.shape
        # NOTE: We want coords to use (x, y) convention
        coords = list(np.ndindex(num_rows, num_cols))
        # Remove any positions that are in the `positions_to_exclude` list.
        if positions_to_exclude:
            logger.debug("Removing `positions_to_exclude` from `coords`...")
            coords = list(set(coords) - set(positions_to_exclude))
    else:
        logger.info(f"Generating {output_size} random fire initial positions...")
        # NOTE: This is very suboptimal when output_size is "close" to sim.fire_map.size.
        while len(coords) < output_size:
            # generate a random coordinate pair within the bounds of the fire map
            # FIXME: Unsure if this will work for rectangular fire maps.
            coord = tuple(
                np.random.randint(low=0, high=sim.fire_map.shape[i], size=(1,)).item()
                for i in range(len(sim.fire_map.shape))
            )
            if coord not in coords and coord not in positions_to_exclude:
                coords.append(coord)

    return coords


@ray.remote
def _simulate_fire_propagation(
    sim: FireSimulation, fire_initial_position: Tuple[int, int]
):
    """"""
    logger.debug(f"Setting fire initial position to: {fire_initial_position}")
    sim.set_fire_initial_position(fire_initial_position)
    # Reset the simulation to ensure the new fire initial position is used.
    sim.reset()

    # Run the simulation until the fire propagation is complete.
    sim.run(1)
    while sim.active:
        sim.run(1)

    burned = int(np.sum((sim.fire_map == BurnStatus.BURNED)))
    unburned = int(np.sum((sim.fire_map == BurnStatus.UNBURNED)))

    logger.debug(f"elapsed_steps == {sim.elapsed_steps}")
    logger.debug(f"total burned squares == {burned}")
    logger.debug(f"total unburned squares == {unburned}\n")

    return {
        "pos_x": fire_initial_position[0],
        "pos_y": fire_initial_position[1],
        "elapsed_steps": sim.elapsed_steps,
        "burned": burned,
        "unburned": unburned,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry-point for generating gifs for different fire start locations.

    Args:
        cfg (DictConfig): Hydra config with all required parameters for training.
    """
    # Start the Ray runtime
    ray.init(address="auto")
    # save_dir = HydraConfig.get().run.dir
    env_cfg = instantiate(cfg.environment.env_config, _convert_="partial")
    sim: FireSimulation = env_cfg["sim"]

    generator_cfg = cfg.simulation.fire_initial_position.generator
    # FIXME: Decide "best" place to call this.
    # Enable `generator` for generating dataset of fire start locations to sample from.
    if generator_cfg:
        start_time = time.time()
        generate_fire_initial_position_data(sim, **generator_cfg)
        end_time = time.time()
        total_runtime = end_time - start_time
        logger.info(f"Total generator runtime: {total_runtime} seconds.")
        logger.info(f"Total generator runtime: {total_runtime/60:.2f} minutes")

    # sampler_cfg = cfg.simulation.fire_initial_position.sampler
    # if sampler_cfg:
    #     pass
    # Enable `sampler` to use for ()


if __name__ == "__main__":
    executed_command = " ".join(["%s" % arg for arg in sys.argv])
    logger.info(f"Executed command: {executed_command}")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    main()
