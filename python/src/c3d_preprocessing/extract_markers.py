from pathlib import Path

from ezc3d import c3d
from numpy.typing import NDArray

from c3d_preprocessing.write_marker_trc_file import write_marker_trc_file


def extract_markers(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    c3d: c3d,
    file: Path,
    input_path: Path,
    rotm: NDArray,
    time_start: float,
    time_end: float,
) -> tuple[int, str]:
    # NOTE: If you index something wrong it does NOT error!
    point_frequency = c3d.parameters.POINT.RATE["value"][0]
    point_units = c3d.parameters.POINT.UNITS["value"][0]
    # Build header
    point_columns = c3d.parameters.POINT.LABELS["value"]
    valid = ["*" not in label for label in point_columns]
    # Get data
    point_columns = [label for label in point_columns if "*" not in label]
    points = c3d.data.points[:, valid, :]
    # TODO(alex): Linear Interpolation  # ruff: ignore[line-contains-todo, missing-todo-link]
    return write_marker_trc_file(
        points,
        point_frequency,
        point_units,
        point_columns,
        input_path,
        file,
        rotm,
        time_start,
        time_end,
    )
