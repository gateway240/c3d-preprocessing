from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def write_marker_trc_file(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    data: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    frequency: float,
    units: str,
    columns: list[str],
    input_path: Path,
    output_path: Path,
    rotm: NDArray,
    time_start: float,  # ruff: ignore[unused-function-argument]
    time_end: float,  # ruff: ignore[unused-function-argument]
) -> tuple[int, str]:
    # print("HI!!")
    # print(data.shape)
    status = 0
    output = ""
    _, num_cols, num_frames = data.shape

    output += "%% ---- Marker PARAMETERS ---- %%\n"
    output += f"Number of frames = {num_frames}\n"
    output += f"Number of columns = {num_cols}\n"
    output += f"Marker frame rate =  {frequency} Hz\n"
    output += f"Marker Units = {units}\n"
    output += f"Marker Names: {columns!s}\n"

    # Write output file
    with output_path.open(encoding="utf-8", mode="w") as fi:
        # Construct the .trc header
        fi.write(f"PathFileType\t4\t(X/Y/Z)\t{input_path!s}\n")
        fi.write(
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
        )
        fi.write(
            f"{frequency:.6f}\t"
            f"{frequency:.6f}\t"
            f"{num_frames:d}\t"
            f"{num_cols:d}\t"
            f"{units}\t"
            f"{frequency:.6f}\t"
            f"{0:d}\t"
            f"{num_frames:d}\n",
        )
        header = "".join(f"{col}\t\t\t" for col in columns)
        fi.write(f"Frame#\tTime\t{header}\n")

        p_idx = list(range(1, num_cols))
        point_xyz_header = "\t".join(f"X{p}\tY{p}\tZ{p}" for p in p_idx)
        fi.write(f"\t\t{point_xyz_header}\n")

        time = np.arange(num_frames) / frequency

        # Slice data if times are valid
        # index_start = 0
        # index_end = num_frames - 1
        # if time_start >= 0.0 && time_end > 0.0
        #     index_start = find(time==time_start);
        #     index_end = find(time==time_end);
        #     output = output + sprintf('Trimming data between times (%f)-(%f)
        #  with indicies [%d]-[%d] \n', ...
        #         time_start,  time_end, index_start, index_end);
        # end

        # Rotate the table
        # Apply the transformation to every point (XYZ combo) for all frames
        # print(rotm.shape)
        # print(data.shape)
        points_out = data.copy()
        points_out[:3] = np.einsum("ij,jkf->ikf", rotm, data[:3])

        # Output the marker file
        for i, t in enumerate(time):
            fi.write(f"{i}\t{t:.16f}\t")
            frame = points_out[:, :, i]
            fi.write(
                "\t".join(
                    # Remap NaNs to OpenSim compatible nan string
                    "nan" if np.isnan(x) else f"{x:.11f}"
                    for x in frame.ravel(order="F")
                ),
            )
            fi.write("\n")

        output += f"[SUCCESS] wrote file: {output_path}\n"

    return status, output
