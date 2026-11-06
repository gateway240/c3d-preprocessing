from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def read_sto_file(  # ruff: ignore[too-many-locals]
    filename: Path,
) -> tuple[NDArray, NDArray, list[str], dict[str, str]]:

    header: dict[str, str] = {}
    values = []
    times = []
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
        # print(labels)
        # print(fi.read())
        rows = fi.read().splitlines()

        for row in rows:
            # print(row)
            row_split = row.split("\t")
            time = float(row_split[0])
            times.append(time)

            data_cols = row_split[1:]
            data_parsed = np.asarray([
                np.fromstring(cell, sep=",", dtype=np.float64) for cell in data_cols
            ])
            values.append(data_parsed)
            # print(data_parsed[0])
            # raise
        # print(times[0])
        # print(values[0])

    times = np.asarray(times)
    values = np.asarray(values)
    # Time col is separate from values so it needs a + 1
    values_cols = np.size(values, axis=1) + 1
    if values_cols % n_columns != 0:
        msg_1 = (
            "Data rows have inconsistent column counts. "
            f"{values_cols} % {n_columns} != 0"
        )
        raise ValueError(msg_1)

    # Convert to (4 or 3, N, T)
    data = values.T
    # print(data[0])
    # print(data.shape)

    return times, data, labels, header
