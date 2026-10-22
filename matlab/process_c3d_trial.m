%% C3D Extraction and Processing
% Ensure the ezc3d library is on the MATLAB path!
clear;
%% Setup - Change these parameters for your own data

% Path Initialization - OS independent
% On Windows® platforms, the file separator character is a backslash (\).
% On other platforms, the file separator might be a different character.
HOME = getenv("HOME");
DATASET_PATH = fullfile(HOME,"data","kuopio-full-body-dataset","s01_raw");
OUTPUT_DIR = fullfile(HOME,"data","kuopio-full-body-dataset-results","s02_extracted");

PARTICIPANT = '09';
MOCAP_SUBDIR = 'mocap';
% name, left foot force plate (fp) index, right foot fp index, start time,
% end time
TRIALS = [
    ["static_cal" 0 0 -1 -1]
    ["jogging" 4 5 0.0 35.0]
    ["walking" 4 5 0.0 25.0]
    ["squats_deep" 3 2 0.0 10.0]
    ["squat_jumps" 3 2 0.0 13.6]
    ];

% Files will be named TRIAL+SUFFIX
MARKERS_SUFFIX = "_markers.trc";
ANALOGS_SUFFIX = "_analog_custom.sto";
GRFS_SUFFIX = "_grfs_custom.sto";
GRFS_RAW_SUFFIX = "_grfs_raw.sto";
GRFS_WRONG_SUFFIX = "_grfs_wrong.sto";

INPUT_PATH = fullfile(DATASET_PATH, PARTICIPANT, MOCAP_SUBDIR);
OUTPUT_PATH = fullfile(OUTPUT_DIR, PARTICIPANT);
if ~exist(OUTPUT_PATH, 'dir')
    mkdir(OUTPUT_PATH)
end

% Mapping from laboratory frame to OpenSim frame
MARKER_ROTATIONS = [-pi/2, pi/2, 0];
% Rotate points to OpenSim frame
% [0,-1,0]
% [0, 0,1]
% [-1,0,0]
rotm = eulang2rotmat(MARKER_ROTATIONS,'XZY');

GRF_CUTOFF = 1; %N

% Low-pass parameters
CUTOFF_FREQUENCY = 6; %Hz
FILTER_ORDER = 2; % forward backward filtering doubles it (so 2 = 4th order)

fprintf("Output Path: %s\n", OUTPUT_PATH);
for i = 1:size(TRIALS,1)
    trial = TRIALS(i,:);
    trial_name = trial(1);
    fp_l = str2double(trial(2));
    fp_r = str2double(trial(3));
    time_start = str2double(trial(4));
    time_end = str2double(trial(5));
    %% C3D Loading
    c3d_file = fullfile(INPUT_PATH,strcat(trial_name,'.c3d'));
    fprintf("Starting to read c3d file: %s\n",c3d_file);
    [c3d, all_pf] = ezc3dRead(convertStringsToChars(c3d_file));
    fprintf("C3D file loaded! Found %d markers and %d force plates!\n", ...
        c3d.parameters.POINT.USED.DATA, length(all_pf));

    %% Marker Extraction - TRIAL_markers.trc file
    marker_file = fullfile(OUTPUT_PATH, strcat(trial_name,MARKERS_SUFFIX));
    fprintf("Starting on file: %s\n", marker_file);
    [status, output] = extractMarkers(c3d, marker_file, INPUT_PATH, rotm, ...
        time_start, time_end);
    fprintf("Status: %d\n%s", status, output);

    % Do not continue processing if we don't have valid force plates
    if fp_l <= 0 || fp_r <= 0
        continue
    end
    %% Analog Extraction - TRIAL_analog.sto file
    analog_file = fullfile(OUTPUT_PATH, strcat(trial_name,ANALOGS_SUFFIX));
    fprintf("Starting on file: %s\n", analog_file);
    [status, output] = extractAnalogs(c3d,analog_file, time_start, time_end);
    fprintf("Status: %d\n%s", status, output);


    %% GRF Extraction - TRIAL_raw.sto file
    grf_file = fullfile(OUTPUT_PATH, strcat(trial_name,GRFS_RAW_SUFFIX));
    fprintf("Starting on file: %s\n", grf_file);
    [status, output] = extractGrfs(c3d, all_pf, grf_file, rotm, ...
        fp_l, fp_r, -1, -1, GRF_CUTOFF, time_start, time_end);
    fprintf("Status: %d\n%s", status, output);
    %% GRF Extraction - TRIAL_grfs.sto file
    grf_file = fullfile(OUTPUT_PATH, strcat(trial_name,GRFS_SUFFIX));
    fprintf("Starting on file: %s\n", grf_file);
    [status, output] = extractGrfs(c3d, all_pf, grf_file, rotm, ...
        fp_l, fp_r, FILTER_ORDER, CUTOFF_FREQUENCY, GRF_CUTOFF, time_start, time_end);
    fprintf("Status: %d\n%s", status, output);
    %% GRF Extraction [WRONG!] - TRIAL_grfs_wrong.sto file
    grf_file = fullfile(OUTPUT_PATH, strcat(trial_name,GRFS_WRONG_SUFFIX));
    fprintf("Starting on file: %s\n", grf_file);
    [status, output] = extractGrfsWrong(c3d, all_pf, grf_file, rotm, ...
        fp_l, fp_r, FILTER_ORDER, CUTOFF_FREQUENCY, GRF_CUTOFF, time_start, time_end);
    fprintf("Status: %d\n%s", status, output);
end
fprintf("Finished all trials!\n");