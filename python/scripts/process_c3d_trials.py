# Logging
import logging
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from ezc3d import c3d

from c3d_preprocessing.extract_analogs import extract_analogs
from c3d_preprocessing.extract_grfs import extract_grfs
from c3d_preprocessing.extract_markers import extract_markers
from c3d_preprocessing.trial import Trial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup - Change these parameters for your own data

# None = Max available for the given platform
MAX_WORKERS = None

# Path Initialization - OS independent
DATASET_PATH = pathlib.Path.home() / "data" / "kuopio-full-body-dataset" / "s01_raw"
OUTPUT_DIR = (
    pathlib.Path.home() / "data" / "kuopio-full-body-dataset-results" / "s02_extracted"
)
# Files will be named TRIAL+SUFFIX
MARKERS_SUFFIX = "_markers.trc"
ANALOGS_SUFFIX = "_analog_custom.sto"
GRFS_SUFFIX = "_grfs_custom.sto"
GRFS_RAW_SUFFIX = "_grfs_raw.sto"
GRFS_WRONG_SUFFIX = "_grfs_wrong.sto"


# Mapping from laboratory frame to OpenSim frame
# MARKER_ROTATIONS = [-pi/2, pi/2, 0];
# Rotate points to OpenSim frame
# [0,-1,0]
# [0, 0,1]
# [-1,0,0]
rotm = np.array([
    [0, -1, 0],
    [0, 0, 1],
    [-1, 0, 0],
])
GRF_CUTOFF = 1  # N

# Low-pass parameters
CUTOFF_FREQUENCY = 6  # Hz
FILTER_ORDER = 2  # forward backward filtering doubles it (so 2 = 4th order)

MOCAP_SUBDIR = "mocap"
# 01...13
PARTICIPANTS = [str(i).zfill(2) for i in range(1, 14)]
# PARTICIPANTS = ["01"]
BASE_TRIALS = [
    ("static_cal", 1, 1),
    ("dyn_sara", 1, 1),
    ("dyn_score_hip", 1, 1),
    ("dyn_score_ankle", 1, 1),
    ("crouch_lift", 3, 2),
    ("crouch_rotate", 3, 2),
    ("curls", 3, 2),
    ("kettlebell", 3, 2),
    ("squats_deep", 3, 2),
    ("half_jacks", 3, 2),
    ("squat_jumps", 3, 2),
    ("box_jabs", 2, 1),
    ("box_combos", 2, 1),
    ("walking", 4, 5),
    ("jogging", 4, 5),
    ("crab_walking", 4, 5),
]
# Force plate overrides
SQUATS_DEEP_SWAP = {"01", "03", "05", "07", "13"}
HALF_JACKS_SWAP = {"01", "02", "03", "05", "07", "13"}
SQUAT_JUMPS_SWAP = {"01", "02", "03", "05", "07", "12", "13"}
BOX_SWAP = {"01", "04"}  # FP3 instead of FP2 for left foot


def get_force_plates(
    participant: str,
    trial: str,
    fp_left: int,
    fp_right: int,
) -> tuple[int, int]:
    if trial == "squats_deep" and participant in SQUATS_DEEP_SWAP:
        return 2, 3

    if trial == "half_jacks" and participant in HALF_JACKS_SWAP:
        return 2, 3

    if trial == "squat_jumps" and participant in SQUAT_JUMPS_SWAP:
        return 2, 3

    if trial in {"box_jabs", "box_combos"} and participant in BOX_SWAP:
        return 3, 1

    return fp_left, fp_right


def process_trial(trial: Trial) -> int:
    status = 0
    trial_name = trial.name
    time_start = trial.start_time
    time_end = trial.end_time
    logger.info("Starting on Participant: %s Trial: %s", trial.participant, trial_name)
    input_path = DATASET_PATH / trial.participant / MOCAP_SUBDIR
    output_path = OUTPUT_DIR / trial.participant
    output_path.mkdir(parents=True, exist_ok=True)

    # C3D Loading
    c3d_file = input_path / f"{trial_name}.c3d"
    logger.info("Starting to read c3d file: %s", c3d_file)
    # NOTE: you cannot pass a pathlib.Path to this function
    c = c3d(str(c3d_file), extract_forceplat_data=True)
    logger.info(
        "C3D file loaded! Found %d markers",
        c["parameters"]["POINT"]["USED"]["value"][0],
    )
    # Marker Extraction - TRIAL_markers.trc file
    marker_file = output_path / f"{trial_name}{MARKERS_SUFFIX}"
    logger.info("Starting on file: %s", marker_file)

    (status, output) = extract_markers(
        c,
        marker_file,
        input_path,
        rotm,
        time_start,
        time_end,
    )
    logger.info("Status: %d\n%s", status, output)

    fp_l = trial.fp_left
    fp_r = trial.fp_right
    # Do not continue processing if we don't have valid force plates
    if fp_l <= 0 or fp_r <= 0:
        return status

    # Analog Extraction - TRIAL_analog.sto file
    analog_file = output_path / f"{trial_name}{ANALOGS_SUFFIX}"
    logger.info("Starting on file: %s", analog_file)

    (status, output) = extract_analogs(
        c,
        analog_file,
        time_start,
        time_end,
    )
    logger.info("Status: %d\n%s", status, output)

    # GRF Extraction - TRIAL_raw.sto file
    grf_raw_file = output_path / f"{trial_name}{GRFS_RAW_SUFFIX}"
    logger.info("Starting on file: %s", grf_raw_file)

    (status, output) = extract_grfs(
        c,
        grf_raw_file,
        rotm,
        fp_l,
        fp_r,
        -1,
        -1,
        GRF_CUTOFF,
        time_start,
        time_end,
    )
    logger.info("Status: %d\n%s", status, output)

    # GRF Extraction - TRIAL_grfs.sto file
    grf_file = output_path / f"{trial_name}{GRFS_SUFFIX}"
    logger.info("Starting on file: %s", grf_file)

    (status, output) = extract_grfs(
        c,
        grf_file,
        rotm,
        fp_l,
        fp_r,
        FILTER_ORDER,
        CUTOFF_FREQUENCY,
        GRF_CUTOFF,
        time_start,
        time_end,
    )
    logger.info("Status: %d\n%s", status, output)
    return status


def main() -> None:
    logger.info("Starting c3d trial processing!")
    trials = [
        Trial(
            participant,
            name,
            *get_force_plates(participant, name, fp_left, fp_right),
            -1,
            -1,
        )
        for participant in PARTICIPANTS
        for name, fp_left, fp_right in BASE_TRIALS
    ]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(process_trial, trials))


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    duration = time.perf_counter() - start
    logger.info("Finished in %.2f seconds", duration)
