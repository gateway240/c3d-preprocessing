from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def read_mot_file(
    filename: Path,
) -> tuple[NDArray, list[str], dict[str, str]]:

    header: dict[str, str] = {}
    # read file
    with filename.open(encoding="utf-8", mode="r") as fi:
        # Read header
        for lin in fi:
            line = lin.strip()

            if line.lower() == "endheader":
                break

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                header[key] = value
        else:
            msg = "Missing endheader marker."
            raise ValueError(msg)

        # Skip blank lines
        for lin in fi:
            line = lin.strip()
            if line:
                break
        else:
            msg_0 = "Missing column header line."
            raise ValueError(msg_0)

        labels = line.split()
        n_columns = len(labels)

        # Read all remaining numeric data
        values = np.fromstring(fi.read(), sep=" ", dtype=np.float64)

    if values.size % n_columns != 0:
        msg_1 = "Data rows have inconsistent column counts."
        raise ValueError(msg_1)

    data_2d = values.reshape((-1, n_columns))

    # Convert to (1, N, T) to match your existing pipeline
    data = data_2d.T[np.newaxis, :, :]

    return data, labels, header
