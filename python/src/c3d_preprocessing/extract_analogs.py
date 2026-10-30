from pathlib import Path

from ezc3d import c3d

from c3d_preprocessing.write_sto_file import write_sto_file


def extract_analogs(
    c3d: c3d,
    file: Path,
    time_start: float,
    time_end: float,
) -> tuple[int, str]:
    # NOTE: If you index something wrong it does NOT error!
    frequency = c3d.parameters.ANALOG.RATE["value"][0]
    # Build header
    columns = c3d.parameters.ANALOG.LABELS["value"]
    # Get data
    points = c3d.data.analogs
    # TODO(alex): Linear Interpolation  # ruff: ignore[line-contains-todo, missing-todo-link]
    return write_sto_file(
        points,
        frequency,
        columns,
        file,
        time_start,
        time_end,
    )
