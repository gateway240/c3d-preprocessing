from typing import Any

import numpy as np
from numpy.typing import NDArray

from c3d_preprocessing.compute_pf_reference_frame import compute_pf_reference_frame


def calculate_fp_grf(  # ruff: ignore[too-many-locals]
    data: NDArray[np.float64],
    fp: Any,  # ruff: ignore[any-type]
    rotm: NDArray,
    grf_cutoff: float,
) -> NDArray[np.float64]:
    _, _num_cols, length_frames = data.shape
    corners = fp["corners"]
    origin = fp["origin"]
    ref_frame = compute_pf_reference_frame(corners)
    mean_corners = np.mean(corners, 1)
    # print(origin)
    # print(ref_frame)
    # print(corners)
    # print(mean_corners)

    # print(data[0,:,0])
    force = data[0, 0:3, :]
    moment = data[0, 3:6, :]

    output = np.zeros((1, 9, length_frames))
    # print(rotm)
    for i in range(length_frames):
        f = force[:, i]
        f_raw = f.copy()
        m = moment[:, i]
        m_raw = m.copy() + np.cross(f, origin)
        fz = f[2]
        valid = -fz >= grf_cutoff

        cop_raw = np.asarray([-m_raw[1] / fz, m_raw[0] / fz, 0])

        f = ref_frame @ f_raw.T
        # m = ref_frame * m_raw';
        cop = ref_frame @ cop_raw.T + mean_corners
        tz = ref_frame @ (m_raw.T - np.cross(f_raw.T, -1 * cop_raw.T))

        if not valid:
            cop = mean_corners
            tz = ref_frame @ (m_raw.T)

        f = rotm @ f
        # m = rotm * m;
        cop = rotm @ cop
        tz = rotm @ tz

        # Convert from mm to m
        cop /= 1000
        tz /= 1000
        # m = m ./1000;

        # Corresponds with OpenSim extract ForceLocation::CenterOfPressure
        # Force 1:3
        # COP 4:6
        # Moment 7:9
        output[:, 0:3, i] = f
        output[:, 3:6, i] = cop
        output[:, 6:9, i] = tz
    return output
