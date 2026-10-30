from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def write_mot_file(
    data: NDArray,
    labels: list[str],
    header: dict[str, str],
    output_path: Path,
) -> tuple[int, str]:
    status = 0
    output = ""
    _, num_cols, num_frames = data.shape

    output += "%% ---- File PARAMETERS ---- %%\n"
    output += f"Number of frames = {num_frames}\n"
    output += f"Number of columns = {num_cols}\n"
    output += f"Columns: {labels!s}\n"

    # Write output file
    with output_path.open(encoding="utf-8", mode="w") as fi:
        # Construct the  header
        fi.write("Coordinates")

        for field, value in header.items():
            fi.write(f"{field}={value}\n")

        fi.write(
            "\nUnits are S.I. units (second, meters, Newtons, ...)\n"
            " If the header above contains a line with 'inDegrees',"
            " this indicates whether rotational values are in degrees (yes) or radians (no).)\n\n",  # ruff: ignore[line-too-long]
        )

        fi.write("endheader\n")

        header_line = "\t".join(labels) + "\t"
        fi.write(f"{header_line}\n")

        for i in range(num_frames):
            frame = data[:, :, i]
            fi.write(f"{frame[0, 0]:.16f}\t")
            fi.write(
                "\t".join(
                    # Remap NaNs to OpenSim compatible nan string
                    "nan" if np.isnan(x) else f"{x:.11f}"
                    for x in frame[:, 1:].ravel(order="F")
                ),
            )
            fi.write("\n")

        output += f"[SUCCESS] wrote file: {output_path}\n"

    return status, output
