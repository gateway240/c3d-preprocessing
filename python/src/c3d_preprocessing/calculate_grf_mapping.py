import numpy as np
from numpy.typing import NDArray


def calculate_grf_mapping(
    analog_columns: list[str],
    fp_l: int,
    fp_r: int,
) -> NDArray:
    # Mapping from analog channel name to GRF name expected in OpenSim
    # [name in file, result key, index]
    fp_l_str = str(fp_l)
    fp_r_str = str(fp_r)
    grf_mapping = np.array([
        [f"Force.Fx{fp_l_str}", "L_ground_force_vx", -1],
        [f"Force.Fy{fp_l_str}", "L_ground_force_vy", -1],
        [f"Force.Fz{fp_l_str}", "L_ground_force_vz", -1],
        ["", "L_ground_force_px", -1],
        ["", "L_ground_force_py", -1],
        ["", "L_ground_force_pz", -1],
        [f"Moment.Mx{fp_l_str}", "L_ground_torque_x", -1],
        [f"Moment.My{fp_l_str}", "L_ground_torque_y", -1],
        [f"Moment.Mz{fp_l_str}", "L_ground_torque_z", -1],
        [f"Force.Fx{fp_r_str}", "R_ground_force_vx", -1],
        [f"Force.Fy{fp_r_str}", "R_ground_force_vy", -1],
        [f"Force.Fz{fp_r_str}", "R_ground_force_vz", -1],
        ["", "R_ground_force_px", -1],
        ["", "R_ground_force_py", -1],
        ["", "R_ground_force_pz", -1],
        [f"Moment.Mx{fp_r_str}", "R_ground_torque_x", -1],
        [f"Moment.My{fp_r_str}", "R_ground_torque_y", -1],
        [f"Moment.Mz{fp_r_str}", "R_ground_torque_z", -1],
    ])

    # Find the indexes for the GRF keys
    for i, analog_label in enumerate(analog_columns):
        for j, row in enumerate(grf_mapping):
            if row[0] == analog_label:
                grf_mapping[j][2] = i
    return grf_mapping
