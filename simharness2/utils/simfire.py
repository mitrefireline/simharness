"""Utilities for parsing `simfire.utils.config.Config` objects."""
import json
import logging
from hashlib import sha256
from typing import Any, Dict

from ray.tune.logger import pretty_print
from ray.tune.utils.util import flatten_dict, unflatten_dict
from simfire.utils.config import Config


logger = logging.getLogger(__name__)


def get_simulator_hash(
    config: Config, return_params_subset: bool = False
) -> Dict[str, Any]:
    """Get unique hash value for the provided `simfire.utils.config.Config` object."""
    output = {}
    logger.debug(f"Generating hash for config:\n {pretty_print(config.yaml_data)}")
    params_subset = parse_config(config)
    if return_params_subset:
        output["params_subset"] = params_subset

    logger.debug(f"Parsed config:\n {pretty_print(params_subset)}")
    output["hash_value"] = _convert_dict_to_hash(params_subset)
    logger.debug(f"Hash value: {output['hash_value']}")
    return output


def parse_config(cfg: Config) -> Dict[str, any]:
    """Parse out parameters from provided `simfire.utils.config.Config` object.

    Any key that starts with "display" is ignored, as these parameters do not affect the
    fire propagation. For keys that start with "simulation", all but the "update_rate"
    and "runtime" keys are ignored (ex: "headless", "sf_home", "save_data", "data_type").

    For the keys starting with "terrain", subkeys that do not match the specified
    topography (fuel) `type` are ignored.

    For the keys starting with "fire", subkeys that contain "pos" are ignored. This is
    because the value of `config.fire.fire_initial_position` will be changed when
    obtaining fire propagation results for various fire start locations.

    For the keys starting with "wind", subkeys that do not match the specified wind
    `function` are ignored.

    TODO: Provide more context on use case (to create hash value for a fire scenario).
    TODO: Move `DELIMITER` to a config (or constants) file.
    TODO: Update hardcoded keys in helper methods to use config (or constants) file.
    """
    DELIMITER = "/"  # TODO: Move to config (or constants) file.
    flat_cfg = flatten_dict(cfg.yaml_data, delimiter=DELIMITER)
    params_subset = {}

    # Area parameters
    params_subset.update(_get_area_params(flat_cfg))
    # Simulation parameters
    params_subset.update(_get_simulation_params(flat_cfg))
    # Mitigation parameters
    params_subset.update(_get_mitigation_params(flat_cfg))
    # Operational parameters
    params_subset.update(_get_operational_params(flat_cfg))
    # Terrain parameters
    params_subset.update(_get_terrain_params(flat_cfg))
    # Fire parameters
    params_subset.update(_get_fire_params(flat_cfg))
    # Environment parameters
    params_subset.update(_get_environment_params(flat_cfg))
    # Wind parameters
    params_subset.update(_get_wind_params(flat_cfg))

    return unflatten_dict(params_subset, delimiter=DELIMITER)


def _get_area_params(flat_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse out desired parameters from the specified `Area` config."""
    return {k: v for k, v in flat_cfg.items() if k.startswith("area")}


def _get_simulation_params(flat_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse out desired parameters from the specified `Simulation` config.

    NOTE: Only the update `update_rate` and `runtime` keys will be included.
    """
    return {
        k: v
        for k, v in flat_cfg.items()
        if k.startswith("simulation") and ("update_rate" in k or "runtime" in k)
    }


def _get_mitigation_params(flat_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse out desired parameters from the specified `Mitigation` config."""
    return {k: v for k, v in flat_cfg.items() if k.startswith("mitigation")}


def _get_operational_params(flat_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse out desired parameters from the specified `Operational` config."""
    return {k: v for k, v in flat_cfg.items() if k.startswith("operational")}


def _get_terrain_params(flat_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse out desired parameters from the specified `Terrain` config.

    NOTE: Only keys for the chosen topo (fuel) `type` are included.
    """
    terrain_params = {k: v for k, v in flat_cfg.items() if k.startswith("terrain")}
    subset_terrain_params = _get_topography_params(terrain_params)
    subset_terrain_params.update(_get_fuel_params(terrain_params))
    return subset_terrain_params


def _get_topography_params(terrain_params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse out desired parameters from the specified `terrain.topography` config."""
    topo_type = [v for k, v in terrain_params.items() if "topo" in k and "type" in k][0]
    return {
        k: v
        for k, v in terrain_params.items()
        if ("topo" in k and "type" in k) or topo_type in k
    }


def _get_fuel_params(terrain_params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse out desired parameters from the specified `terrain.fuel` config."""
    fuel_type = [v for k, v in terrain_params.items() if "fuel" in k and "type" in k][0]
    return {
        k: v
        for k, v in terrain_params.items()
        if ("fuel" in k and "type" in k) or fuel_type in k
    }


def _get_fire_params(flat_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse out desired parameters from the specified `Fire` config."""
    return {k: v for k, v in flat_cfg.items() if k.startswith("fire") and "pos" not in k}


def _get_environment_params(flat_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse out desired parameters from the specified `Environment` config."""
    return {k: v for k, v in flat_cfg.items() if k.startswith("environment")}


def _get_wind_params(flat_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse out desired parameters from the specified `Wind` config."""
    wind_func = [v for k, v in flat_cfg.items() if "wind" in k and "func" in k][0]
    return {
        k: v
        for k, v in flat_cfg.items()
        if ("wind" in k and "func" in k) or wind_func in k
    }


def _convert_dict_to_hash(config: Dict[str, Any]) -> str:
    """Convert a dictionary to a unique hash value.

    This implementation first converts the dictionary into a JSON string using
    `json.dumps()`, ensuring that keys are sorted alphabetically so that different
    orderings produce the same output. It then encodes the string as bytes before passing
    it to the SHA256 hash function provided by Python's built-in "hashlib' module. The
    resulting hexadecimal digest represents a unique id for the dictionary contents.
    """
    config_str = json.dumps(config, sort_keys=True).encode("utf8")
    return sha256(config_str).hexdigest()
