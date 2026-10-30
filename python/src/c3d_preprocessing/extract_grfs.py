from pathlib import Path

import numpy as np
from ezc3d import c3d
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt

from c3d_preprocessing.calculate_fp_grf import calculate_fp_grf
from c3d_preprocessing.calculate_grf_mapping import calculate_grf_mapping
from c3d_preprocessing.write_sto_file import write_sto_file


def fillmissing_linear(arr: NDArray) -> NDArray:
    out = arr.copy()

    _, num_cols, _ = out.shape

    for col in range(num_cols):
        y = out[0, col, :]
        valid = ~np.isnan(y)

        if valid.any():
            y[~valid] = np.interp(
                np.flatnonzero(~valid),
                np.flatnonzero(valid),
                y[valid],
            )

    return out


def extract_grfs(  # ruff: ignore[too-many-arguments, too-many-locals, too-many-positional-arguments]
    c3d: c3d,
    output_file: Path,
    rotm: NDArray,
    fp_l: int,
    fp_r: int,
    filter_order: float,
    filter_cutoff_frequency: float,
    force_minimum: float,
    time_start: float,
    time_end: float,
) -> tuple[int, str]:
    # NOTE: If you index something wrong it does NOT error!
    frequency = c3d.parameters.ANALOG.RATE["value"][0]
    # Build header
    columns = c3d.parameters.ANALOG.LABELS["value"]
    # Get data
    analogs = np.asarray(c3d.data.analogs)
    _, _num_cols, num_frames = analogs.shape
    all_pf = c3d.data["platform"]
    grf_mapping = calculate_grf_mapping(columns, fp_l, fp_r)
    # print(grf_mapping)
    grf_columns = grf_mapping[:, 1].tolist()
    # Low-pass filter raw signals
    fc = filter_cutoff_frequency
    fs = frequency
    # Second order filter forwards-backwards is effectively fourth order
    n = filter_order
    # Only filter if filter parameters are valid
    a = None
    b = None
    if fc >= 0 and n > 0:
        # From Winter Biomechanics and Motor Control p. 69
        c = (2 ** (1 / 2) - 1) ** (1 / (2 * n))
        fc_corrected = fc / c

        wn = fc_corrected / (fs / 2)

        [b, a] = butter(n, wn, "low")

    fp_active = [fp_l, fp_r]

    num_inputs = sum(str(fp_l) in str(label) for label in grf_mapping[:, 0])
    grf_mapping_len, _grf_mapping_cols = grf_mapping.shape
    num_outputs = grf_mapping_len // len(fp_active)
    # print("fp_l =", fp_l)
    # print("fp_active =", fp_active)
    # print("num_inputs =", num_inputs)
    # print("grf_mapping[:,0] =", grf_mapping[:,0])
    # print(analogs.shape)

    grf_filtered = np.zeros(
        (1, num_inputs * len(fp_active), num_frames),
    )
    grf_index = 0
    for i in range(grf_mapping_len):
        index = int(grf_mapping[i, 2])
        if index >= 0:
            # Only filter if filter parameters are valid
            if fc >= 0 and n > 0 and a and b:
                grf_filtered[:, grf_index, :] = filtfilt(b, a, analogs[:, index, :])
            else:
                grf_filtered[:, grf_index, :] = analogs[:, index, :]
            grf_index += 1

    # Re-calculate COP from analog force and moment signal
    grfs_final = np.zeros((1, grf_mapping_len, num_frames))
    for j, fp_index in enumerate(fp_active):
        # We use the plate index while the 0 index is -1
        fp = all_pf[fp_index - 1]

        stride_in = j * num_inputs
        data = grf_filtered[:, stride_in : stride_in + num_inputs, :]

        stride_out = j * num_outputs
        grf_result = calculate_fp_grf(data, fp, rotm, force_minimum)
        # Fill NaNs on the columns - OpenSim ID can't handle NaN values
        # final_result = fillmissing_linear(grf_result)
        # grfs_final: (1, N, T)
        # print("final_result.shape", final_result.shape)
        # print("num_outputs", num_outputs)
        grfs_final[
            :,
            stride_out : stride_out + num_outputs,
            :,
        ] = grf_result

    # Write the sto file
    return write_sto_file(
        grfs_final,
        frequency,
        grf_columns,
        output_file,
        time_start,
        time_end,
    )
