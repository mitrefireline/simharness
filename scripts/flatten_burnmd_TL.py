import os
import sys
import json
import hydra
from omegaconf import DictConfig
import time
import logging

if f"{os.environ['HOME']}/simharness" not in sys.path:
    sys.path.append(os.path.join(os.environ["HOME"], "simharness2"))

logger = logging.getLogger(__name__)

# List of substrings that indicate a bad location in the BurnMD data.
BAD_LOCATIONS = ["Oregon_2021"]


@hydra.main(
    version_base=None,
    config_path=f"{os.environ['HOME']}/simharness/conf/scripts",
    config_name="flatten_burnmd_TL",
)
def main(cfg: DictConfig) -> None:
    """Convert BurnMD data to a flat format to make sampling easier.

    The BurnMD TL (Top Left) data is provided in a nested format, where the outermost key
    is state (only continental US), then year, then fire name, and finally the location,
    which represents the top left bound of the fire perimeter, in (lat, lon) format.

    This function flattens the data to make it easier to sample from, by converting the


    """
    # Load unflattened BurnMD data from provided input path.
    logger.info(f"Loading BurnMD data from {cfg.burnmd.input_path}")
    with open(cfg.burnmd.input_path, "r", encoding="utf-8") as f:
        burnmd_op_locs = json.loads(f.read())

    # Flatten BurnMD data
    flat_burnmd_op_locs = {}
    unique_fires = 0
    logger.info("Flattening BurnMD data...")
    start_time = time.time()

    # Iterate over the nested BurnMD data and flatten it.
    for state, state_data in burnmd_op_locs.items():
        for year, year_data in state_data.items():
            for fire_name, lat_lon in year_data.items():
                # Process lat_lon string into a tuple of floats.
                lat_lon = tuple(map(float, lat_lon.strip("()").split(",")))
                # Create a unique key for each fire.
                fire_name = fire_name.replace(" ", "_")
                key = f"{state}_{year}_{fire_name}"
                # Skip bad locations.
                if any(bad_loc in key for bad_loc in BAD_LOCATIONS):
                    logger.warning(f"Skipping bad location: {key}")
                    continue
                logger.debug(f"Processing fire: {key}")
                flat_burnmd_op_locs[key] = {
                    "state": state,
                    "year": int(year),
                    "fire_name": fire_name,
                    "latitude": lat_lon[0],
                    "longitude": lat_lon[1],
                }
                unique_fires += 1

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Flattened {unique_fires} unique fires in {total_time:.6f} seconds.")

    # NOTE: This assertion can be removed, but it's a good sanity check to ensure that
    # there is no data loss during the flattening process.
    assert len(flat_burnmd_op_locs) == unique_fires

    # Save flattened BurnMD data to provided output path.
    logger.info(f"Saving flattened BurnMD data to {cfg.burnmd.output_path}")
    with open(cfg.burnmd.output_path, "w", encoding="utf-8") as f:
        json.dump(flat_burnmd_op_locs, f, indent=4)


if __name__ == "__main__":
    main()
