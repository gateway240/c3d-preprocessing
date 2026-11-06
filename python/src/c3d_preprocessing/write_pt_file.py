import c3d_preprocessing.global_rotations
from c3d_preprocessing.global_rotations import global_rotations
from c3d_preprocessing.armature import SMPLHJoint
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

# From the transposer manuscript
SENSOR_MAPPING = [
    "pelvis_imu",  # 6. pelvis (root)
    "tibia_l_imu",  # 3. left lower leg
    "tibia_r_imu",  # 4. right lower leg
    "head_imu",  # 5. head imu
    "radius_l_imu",  # 1. left forearm
    "radius_r_imu",  # 2. right forearm
]
tracked_joints = [
    SMPLHJoint.PELVIS,
    SMPLHJoint.LKNEE,
    SMPLHJoint.RKNEE,
    SMPLHJoint.HEAD,
    SMPLHJoint.LELBOW,
    SMPLHJoint.RELBOW,
]

def make_calibration_pose(num_frames: int=1) -> torch.Tensor:
    pose = torch.eye(3).repeat(num_frames, 24, 1, 1)

    left_arm_down = torch.tensor(
        Rotation.from_euler("z", -85, degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    right_arm_down = torch.tensor(
        Rotation.from_euler("z", 85, degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    left_elbow_bend = torch.tensor(
        Rotation.from_euler("y", -10, degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    right_elbow_bend = torch.tensor(
        Rotation.from_euler("y", 10, degrees=True).as_matrix(),
        dtype=torch.float32,
    )

    pose[:, SMPLHJoint.LSHOULDER] = left_arm_down
    pose[:, SMPLHJoint.RSHOULDER] = right_arm_down

    pose[:, SMPLHJoint.LELBOW] = left_elbow_bend
    pose[:, SMPLHJoint.RELBOW] = right_elbow_bend

    return pose

def calibrate_rotations(
    imu_rotations,   # [frames,6,3,3]
    pose,            # calibration pose
):
    parent = [
        -1, 0, 0, 0,
         1, 2, 3,
         4, 5, 6,
         7, 8, 9,
        12, 12, 12,
        13, 14,
        16, 17,
        18, 19,
        20, 21,
    ]

    tracked_joints = [
        SMPLHJoint.PELVIS,
        SMPLHJoint.LKNEE,
        SMPLHJoint.RKNEE,
        SMPLHJoint.HEAD,
        SMPLHJoint.LELBOW,
        SMPLHJoint.RELBOW,
    ]

    # calibration pose bone orientations
    g_rot = torch.tensor(global_rotations, dtype=torch.float32)

    R_bone_M_calib = g_rot[0, tracked_joints]

    # average calibration orientation
    R_sensor_I_calib = imu_rotations[0:60].mean(0)

    # Eq. 15
    P_IM = torch.eye(
        3,
        dtype=torch.float32,
    )

    # Eq. 18
    R_offset_I = (
        R_sensor_I_calib.transpose(-1, -2)
        @ P_IM.unsqueeze(0)
        @ R_bone_M_calib
    )

    # runtime reconstruction
    # Eq. 16 + Eq. 17
    R_bone_M = (
        P_IM.T
        .unsqueeze(0)
        .unsqueeze(0)
        @ (
            imu_rotations
            @ R_offset_I.unsqueeze(0)
        )
    )

    return R_bone_M

def write_pt_file(
    data: NDArray,
    columns: list[str],
    output_path: Path,
) -> tuple[int, str]:
    status = 0
    output = ""
    unit_size, num_cols, num_frames = data.shape

    # drop time
    columns = columns[1:]
    output += "%% ---- File PARAMETERS ---- %%\n"
    output += f"Number of frames = {num_frames}\n"
    output += f"Number of columns = {num_cols}\n"
    output += f"Unit size = {unit_size}\n"
    output += f"Columns: {columns!s}\n"

    valid_sensors = np.asarray([
        i for i, val in enumerate(columns) if val in SENSOR_MAPPING
    ])
    # print(valid_sensors)
    # print(len(valid_sensors))
    data_slice = np.take(data, valid_sensors, axis=1)
    _, num_sensors, _ = data_slice.shape
    # data_slice = data[:, valid_sensors.keys(), :]
    # print(data_slice.shape)
    # CALIBRATION
    # g_rot = torch.tensor(global_rotations, dtype=torch.float32)
    # print(g_rot.shape)
    # R_bone_M = torch.stack(
    #     [g_rot[0, j] for j in tracked_joints]
    # )
    pose = make_calibration_pose()

    if unit_size == 3:
        tensor = torch.from_numpy(data_slice.T).to(torch.float32)
        # calibrate_accelerations()
    # print("No change for Accelerations!")
    if unit_size == 4:  # ruff: ignore[magic-value-comparison]
        # print("Converting orientations!")
        data_slice_quat = np.empty([3, 3, num_sensors, num_frames])
        for sensor in range(num_sensors):
            qs = data_slice[:, sensor, :].T  # (num_frames, 4)
            # Osim uses w,x,y,z format
            mats = (
                Rotation.from_quat(
                    qs,
                    scalar_first=True,
                ).as_matrix()  # (num_frames, 3, 3)
            )

            data_slice_quat[:, :, sensor, :] = mats.transpose(1, 2, 0)
        data_slice = data_slice_quat
        d = torch.from_numpy(data_slice.T).to(torch.float32)
        # print(data_slice_quat.shape)
        R_bone_M = calibrate_rotations(
            d,   # calibration segment
            pose,
        )
        print(R_bone_M.shape)
 
        # overwrite with calibrated data
        tensor = R_bone_M
        # print(tensor.shape)


    output += f"Sensors: {SENSOR_MAPPING} \n"
    output += f"Selected indices: {valid_sensors}\n"
    output += f"Writing tensor with shape {tensor.shape}\n"
    # print(tensoRotation.shape)
    # Write output file
    torch.save(tensor, output_path)
    # with output_path.open(encoding="utf-8", mode="w") as fi:
    #     # Construct the .trc header
    #     fi.write(f"nColumns={(num_cols + 1):d}\n")  # +1 for time
    #     fi.write(f"nRows={num_frames:d}\n")
    #     fi.write("DataType=double\n")
    #     fi.write("version=3\n")
    #     fi.write("OpenSimVersion=4.6\n")
    #     fi.write("endheader\n")
    #     header = "".join(f"{col}\t" for col in columns)
    #     fi.write(f"time\t{header}\n")

    #     # time = np.arange(num_frames) / frequency

    #     # Slice data if times are valid
    #     # index_start = 0
    #     # index_end = num_frames - 1
    #     # if time_start >= 0.0 && time_end > 0.0
    #     #     index_start = find(time==time_start);
    #     #     index_end = find(time==time_end);
    #     #     output = output + sprintf('Trimming data between times (%f)-(%f)
    #     #  with indicies [%d]-[%d] \n', ...
    #     #         time_start,  time_end, index_start, index_end);
    #     # end

    #     # Rotate the table
    #     # Apply the transformation to every point (XYZ combo) for all frames
    #     # print(rotm.shape)
    #     # print(data.shape)
    #     # Please note that the shape of point_data is 4xNxT,
    #     # where 4 represent the components XYZ1
    #     # (the 3D coordinates of the point add with a 1 )
    #     points_out = data.copy()
    #     points_out[:3] = np.einsum("ij,jkf->ikf", rotm, data[:3])
    #     # Output the marker file
    #     for i, t in enumerate(time):
    #         fi.write(f"{t:.16f}\t")
    #         frame = points_out[:, :, i]
    #         fi.write(
    #             "\t".join(
    #                 # Remap NaNs to OpenSim compatible nan string
    #                 "nan" if np.isnan(x) else f"{x:.11f}"
    #                 for x in frame.ravel(order="F")
    #             ),
    #         )
    #         fi.write("\n")

    output += f"[SUCCESS] wrote file: {output_path}\n"

    return status, output
