import os
import sys
import json
import hydra
from omegaconf import DictConfig
import time
import logging
from shapely.geometry import Point, Polygon

if f"{os.environ['HOME']}/simharness" not in sys.path:
    sys.path.append(os.path.join(os.environ["HOME"], "simharness2"))

logger = logging.getLogger(__name__)

# List of substrings that indicate a bad location in the BurnMD data.
BAD_LOCATIONS = [
    "Oregon_2021",
    "Colorado_2021_Morgan_Creek",
    "'New Mexico_2019_Rawhide",
]


def validate_coordinates(lon: float, lat: float) -> bool:
    """Validate a lat, lon point can be located within WGS84 CRS.

    This function validates a lat, lon point by checking if it is within the bounds of
    the WGS84 CRS, after wrapping the longitude value within [-180, 180).

    The function returns True if the point is valid, False otherwise.
    Credit to https://gis.stackexchange.com/a/378885.

    Args:
        lon: The longitude value.
        lat: The latitude value.

    Returns:
        True if the point is valid, False otherwise.
    """
    # Put the longitude in the range of [0,360):
    lon %= 360
    # Put the longitude in the range of [-180,180):
    if lon >= 180:
        lon -= 360
    lon_lat_point = Point(lon, lat)
    lon_lat_bounds = Polygon.from_bounds(xmin=-180.0, ymin=-90.0, xmax=180.0, ymax=90.0)
    return lon_lat_bounds.intersects(lon_lat_point)


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
                # Validate the lat_lon point.
                valid_lat_lon = validate_coordinates(lon=lat_lon[1], lat=lat_lon[0])
                if not valid_lat_lon:
                    logger.warning(
                        f"Skipping {key} for invalid lat, lon point: {lat_lon[0], lat_lon[1]}"
                    )
                    continue
                # Skip 'bad' locations.
                elif any(bad_loc in key for bad_loc in BAD_LOCATIONS):
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
