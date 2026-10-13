function [status, output] = extractGrfs(c3d, all_pf, output_file, rotm, fp_l, fp_r, ...
    filter_order, filter_cutoff_frequency, force_minimum, time_start, time_end)
%extractMarkers  Marker Extraction - TRIAL_markers.trc file
%   Detailed explanation goes here
arguments (Input)
    c3d
    all_pf
    output_file
    rotm
    fp_l
    fp_r
    filter_order
    filter_cutoff_frequency
    force_minimum
    time_start
    time_end
end

arguments (Output)
    status
    output
end

analog_frequency = c3d.parameters.ANALOG.RATE.DATA;
% Build header
analog_columns = string(c3d.parameters.ANALOG.LABELS.DATA);
% Get data
analogs = c3d.data.analogs;
grf_mapping = calculateGrfMapping(analog_columns, fp_l, fp_r);
grf_columns = grf_mapping(:,2);

% Low-pass filter raw signals
% THIS doesn't work :(
% normalized_cutoff = 6 / (analog_frequency / 2);
% [b, a] = butter(FILTER_ORDER, normalized_cutoff, 'low');

% This works
Fc = filter_cutoff_frequency;
Fs = analog_frequency;
% Second order filter forwards-backwards is effectively fourth order
n = filter_order;
% Only filter if filter parameters are valid
if Fc >= 0 && n > 0
    % From Winter Biomechanics and Motor Control p. 69
    C = (2^(1/2)-1)^(1/(2*n));

    Fc_corrected = Fc/C;

    Wn = Fc_corrected/(Fs/2);

    [b,a] = butter(n,Wn,'low');
end

fp_active = [fp_l, fp_r];
num_inputs = sum(count(grf_mapping(:,1),string(fp_l)));
num_outputs = length(grf_mapping)/ length(fp_active);
grf_filtered = zeros(length(analogs), num_inputs * length(fp_active));

grf_index = 1;
for i = 1:length(grf_mapping)
    index = str2double(grf_mapping(i,3));
    if index > 0
        % Only filter if filter parameters are valid
        if Fc >= 0 && n > 0
            grf_filtered(:,grf_index) = filtfilt(b, a, analogs(:,index));
        else
            grf_filtered(:,grf_index) = analogs(:,index);
        end
        grf_index = grf_index + 1;
    end
end

% Re-calculate COP from analog force and moment signal
grfs_final = zeros(length(analogs), length(grf_mapping));
for j = 1:numel(fp_active)
    fp_index = fp_active(j);
    fp = all_pf(fp_index);
    stride_in = (j-1)*num_inputs;

    data = grf_filtered(:, 1+stride_in: num_inputs+stride_in);

    stride_out =  (j-1)*num_outputs;
    grf_result = calculateFpGrf(data,fp,rotm,force_minimum);
    % Fill NaNs on the columns - OpenSim ID can't handle NaN values
    final_result = fillmissing(grf_result,"linear",1);
    grfs_final(:, 1+stride_out: num_outputs+stride_out) = final_result;

end
% Write the sto file
[status, output] = writeStoFile(grfs_final,analog_frequency, ...
    grf_columns,output_file, time_start, time_end);
end