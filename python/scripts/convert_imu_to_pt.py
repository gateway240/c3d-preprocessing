# Logging
import logging
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from c3d_preprocessing.read_sto_file import read_sto_file
from c3d_preprocessing.trial import Trial
from c3d_preprocessing.write_pt_file import write_pt_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup - Change these parameters for your own data

# None = Max available for the given platform
MAX_WORKERS = None

# Path Initialization - OS independent
DATASET_PATH = (
    pathlib.Path.home() / "data" / "kuopio-full-body-dataset-results" / "s03_aligned"
)
OUTPUT_DIR = (
    pathlib.Path.home() / "data" / "kuopio-full-body-dataset-results" / "s02_aligned_pt"
)


# Low-pass parameters
CUTOFF_FREQUENCY = 6  # Hz
FILTER_ORDER = 2  # forward backward filtering doubles it (so 2 = 4th order)

# 01...13
# PARTICIPANTS = [str(i).zfill(2) for i in range(1, 14)]
PARTICIPANTS = ["01"]
BASE_TRIALS = [
    "static_cal",
    "dyn_sara",
    "dyn_score_hip",
    "dyn_score_ankle",
    "crouch_lift",
    "crouch_rotate",
    "curls",
    "kettlebell",
    "squats_deep",
    "half_jacks",
    "squat_jumps",
    "box_jabs",
    "box_combos",
    "walking",
    "jogging",
    "crab_walking",
]
# TRIAL_SUFFIX = "marker_ik_output.mot"
EXT = ".sto"
MODALITIES = [
    "_accelerations.sto",
    "_orientations.sto",
]


def process_trial(trial: Trial) -> int:  # ruff: ignore[too-many-locals]
    status = 0
    trial_name = trial.name
    logger.info("Starting on Participant: %s Trial: %s", trial.participant, trial_name)
    input_path = DATASET_PATH / trial.participant
    output_path = OUTPUT_DIR / trial.participant
    output_path.mkdir(parents=True, exist_ok=True)

    filtered_suffix = ".pt"

    dir_data = list(input_path.glob(f"*{EXT}"))
    # print(dir_data)
    # Trial Loading
    matching_files = [
        f
        for f in dir_data
        for modality in MODALITIES
        if f.name.endswith(modality) and trial_name in f.name
    ]
    # print(matching_files)
    status = 0
    output = ""
    for data_file in matching_files:
        logger.info("Starting to read file: %s", data_file)
        (time, data, labels, _header) = read_sto_file(
            data_file,
        )
        # Trial filtering
        output_full_name = f"{data_file.stem}{filtered_suffix}"
        output_file = output_path / output_full_name
        logger.info("Starting on file: %s\n", output_file)

        step = float(time[1] - time[0])
        tf = np.allclose(np.diff(time), step)
        logger.info("Time step uniform: %s", tf)

        # Start filtering
        # Low-pass filter raw signals
        # fc = CUTOFF_FREQUENCY
        # fs = 1 / step
        # # Second order filter forwards-backwards is effectively fourth order
        # n = FILTER_ORDER
        # # From Winter Biomechanics and Motor Control p. 69
        # c = (2 ** (1 / 2) - 1) ** (1 / (2 * n))
        # fc_corrected = fc / c

        # wn = fc_corrected / (fs / 2)

        # [b, a] = butter(n, wn, "low")

        # # Filter all columns except time
        # data[:, 1:, :] = filtfilt(b, a, data[:, 1:, :])

        # [status, output] = writeMotFile(data, labels, header, output_file);
        (status, output) = write_pt_file(
            data,
            labels,
            output_file,
        )

    logger.info("Status: %d\n%s", status, output)
    return status


def main() -> None:
    logger.info("Starting c3d trial processing!")
    trials = [
        Trial(
            participant,
            name,
            -1,
            -1,
            -1,
            -1,
        )
        for participant in PARTICIPANTS
        for name in BASE_TRIALS
    ]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(process_trial, trials))


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    duration = time.perf_counter() - start
    logger.info("Finished in %.2f seconds", duration)
