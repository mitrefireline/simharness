from typing import Tuple

import numpy as np


def get_adjacent_points(
    row: int, col: int, shape: Tuple[int, int], include_diagonals: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Return points adjacent to the provided point (excluding point itself).

    The current implementation considers the 4 cardinal directions (N, S, E, W) as
    adjacent points. If `include_diagonals` is set to True, the diagonal points
    (NE, NW, SE, SW) are also considered as adjacent points.

    Arguments:
        row: The row index of the current point.
        col: The column index of the current point.
        shape: A 2-tuple representing the shape of the map.
        include_diagonals: A boolean indicating whether to include diagonal points
            as adjacent points. Defaults to True.

    Returns:
        A tuple containing two numpy arrays, adjacent rows and adjacent columns. The
        returned arrays can be used as an advanced index to access the adjacent
        points, ex: `fire_map[adj_rows, adj_cols]`.
    """
    # TODO: Logic below is copied from a method in simfire, namely
    # simfire.game.managers.fire.FireManager._get_new_locs(). It would be good to
    # refactor this logic into a utility function in simfire, and then call it here.
    x, y = col, row
    # Generate all possible adjacent points around the current point.
    if include_diagonals:
        new_locs = (
            (x + 1, y),
            (x + 1, y + 1),
            (x, y + 1),
            (x - 1, y + 1),
            (x - 1, y),
            (x - 1, y - 1),
            (x, y - 1),
            (x + 1, y - 1),
        )
    else:
        new_locs = (
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
            (x, y - 1),
        )

    col_coords, row_coords = zip(*new_locs)
    adj_array = np.array([row_coords, col_coords], dtype=np.int32)

    # Clip the adjacent points to ensure they are within the map boundaries
    row_max, col_max = [dim - 1 for dim in shape]
    adj_array = np.clip(adj_array, a_min=[[0], [0]], a_max=[[row_max], [col_max]])
    # Remove the point itself from the list of adjacent points, if it exists.
    adj_array = adj_array[:, ~np.all(adj_array == [[row], [col]], axis=0)]

    return adj_array[0], adj_array[1]
