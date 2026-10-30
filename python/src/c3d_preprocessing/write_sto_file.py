from pathlib import Path

import numpy as np


def write_sto_file(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    data: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    frequency: float,
    columns: list[str],
    output_path: Path,
    time_start: float,  # ruff: ignore[unused-function-argument]
    time_end: float,  # ruff: ignore[unused-function-argument]
) -> tuple[int, str]:
    status = 0
    output = ""
    _, num_cols, num_frames = data.shape

    output += "%% ---- File PARAMETERS ---- %%\n"
    output += f"Number of frames = {num_frames}\n"
    output += f"Number of columns = {num_cols}\n"
    output += f"Frame rate =  {frequency} Hz\n"
    output += f"Columns: {columns!s}\n"

    # Write output file
    with output_path.open(encoding="utf-8", mode="w") as fi:
        # Construct the .trc header
        fi.write(f"DataRate={frequency:.6f}\n")
        fi.write(f"nColumns={(num_cols + 1):d}\n")  # +1 for time
        fi.write(f"nRows={num_frames:d}\n")
        fi.write("DataType=double\n")
        fi.write("version=3\n")
        fi.write("OpenSimVersion=4.6\n")
        fi.write("endheader\n")
        header = "".join(f"{col}\t" for col in columns)
        fi.write(f"time\t{header}\n")

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
        # Please note that the shape of point_data is 4xNxT,
        # where 4 represent the components XYZ1
        # (the 3D coordinates of the point add with a 1 )
        points_out = data.copy()
        # points_out[:3] = np.einsum("ij,jkf->ikf", rotm, data[:3])
        # Output the marker file
        for i, t in enumerate(time):
            fi.write(f"{t:.16f}\t")
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
