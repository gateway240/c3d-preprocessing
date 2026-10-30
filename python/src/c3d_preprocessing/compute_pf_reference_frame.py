import numpy as np
from numpy.typing import NDArray


def compute_pf_reference_frame(corners: NDArray) -> NDArray:
    # Same as ezc3d computePfReferenceFrame
    axis_x = corners[:, 0] - corners[:, 1]
    axis_y = corners[:, 0] - corners[:, 3]
    axis_z = np.cross(axis_x, axis_y)
    axis_y = np.cross(axis_z, axis_x)

    # Same normalization as ezc3d
    # axis_x = axis_x / sqrt[dot[axis_x,axis_x];
    axis_x /= np.linalg.norm(axis_x)
    axis_y /= np.linalg.norm(axis_y)
    axis_z /= np.linalg.norm(axis_z)

    return np.vstack((axis_x[:].T, axis_y[:].T, axis_z[:].T))
