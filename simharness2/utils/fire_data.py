"""FIXME: A one line summary of the module or program.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
import os
import logging
from typing import List, Tuple, Dict
import yaml

import modin.pandas as pd
import ray
import numpy as np

from simfire.sim.simulation import FireSimulation
from simfire.enums import BurnStatus
from simharness2.utils.simfire import get_simulator_hash


logger = logging.getLogger(__name__)

# Constants. TODO: Move to `constants.py` file.
FIRE_DATA_DTYPES = {
    "x": np.uint16,
    "y": np.uint16,
    "elapsed_steps": np.uint16,
    "burned": np.uint32,
    "unburned": np.uint32,
    "percent_area_burned": np.float16,
}


def filter_fire_initial_position_data(
    *,
    fire_df: pd.DataFrame,
    sample_size: Dict[str, int],
    query: str,
    population_size: int = None,
    **kwargs,
) -> Tuple[np.recarray, np.recarray]:
    """TODO"""
    eval_size = sample_size.get("eval")
    train_df, eval_df = pd.DataFrame({}), pd.DataFrame({})
    logger.info(f"Applying the following condition to `fire_df`: {query}")
    subset_fire_df: pd.DataFrame = fire_df.query(query)

    # Compute absolute minimum positions needed for valid sampling. Recall that when
    # `population_size` is None, the population is set to all remaining rows.
    if population_size is None:
        min_pos_needed = sum([v for v in sample_size.values()])
    else:
        min_pos_needed = sum([eval_size, population_size])

    num_pos = len(subset_fire_df)
    logger.info(f"There are {num_pos} positions after applying the condition.")

    # Ensure that the filtered "dataset" is large enough to sample from.
    if num_pos < min_pos_needed:
        msg = (
            f"There must be AT LEAST {min_pos_needed} positions to enable sampling. "
            f"There are only {num_pos} positions when using the condition: {query}. "
            "Try relaxing the `query` conditions, or decrease the `population_size`."
        )
        raise ValueError(msg)

    # Extract data to be used for evaluation.
    # TODO: Allow for user-provided evaluation dataset?
    if eval_size:
        eval_df: pd.DataFrame = subset_fire_df.sample(n=eval_size, replace=False)
        # Ensure the evaluation data cannot be sampled again (ie for training data).
        train_df: pd.DataFrame = subset_fire_df.drop(eval_df.index)
        # Downsample the training data to have exactly `population_size` total samples.
        if population_size:
            train_df: pd.DataFrame = train_df.sample(n=population_size, replace=False)
    else:
        # TODO: hydra should ENFORCE the existence of the `sample_size.eval` key.
        raise ValueError("`sample_size.eval` is required!")

    logger.info(f"The eval dataset contains {len(eval_df)} samples after processing.")
    logger.info(f"The train dataset contains {len(train_df)} samples after processing.")

    # Convert filtered train/eval "dataset" to a structured NumPy array (for zero-copy).
    train_arr = train_df.to_records(index=False)
    eval_arr = eval_df.to_records(index=False)

    return train_arr, eval_arr


def generate_fire_initial_position_data(
    sim: FireSimulation,
    save_path: str,
    output_size: int = 1,
    make_all_positions: bool = False,
) -> pd.DataFrame:
    """Generate fire initial position data for a given fire simulation.

    Args:
        sim: A `FireSimulation` instance.
        save_path: A `str` specifying the path to save the generated data.
        output_size: An `int` representing the number of fire initial positions to
            generate. Defaults to 1.
        make_all_positions: A `bool` indicating whether to generate all possible fire
            initial positions, or only `output_size` positions. Defaults to False.
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
    data_save_path = os.path.join(save_path, "fire_initial_positions.csv")
    # TODO: Move this if block into `load_fire_initial_position_data` (or similar).
    # Check if the fire initial position data has already been generated.
    if os.path.exists(data_save_path):
        logger.info(
            f"Fire initial position data for {sim_hash_value} has already been generated."
        )
        pos_df = pd.read_csv(data_save_path, dtype=FIRE_DATA_DTYPES)
        total_pos = sim.fire_map.size if make_all_positions else output_size
        missing_pos = total_pos - len(pos_df)
        # Ensure all positions have been generated, and if not, generate them.
        if missing_pos > 0:
            logger.warning(
                f"Expected {output_size} positions, but only found {len(coords)}."
            )
            curr_pos = [tuple(row) for row in pos_df[["x", "y"]].values]
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
            # TODO: What should we return? We can put the object in shared object store
            # and return object id. Or, we can return the data directly. Any others?
            return pos_df

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
        logger.info(f"Saving fire initial position data to: {data_save_path}")
        df.to_csv(data_save_path, index=False)

        return df


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
        "x": fire_initial_position[0],
        "y": fire_initial_position[1],
        "elapsed_steps": sim.elapsed_steps,
        "burned": burned,
        "unburned": unburned,
    }
