function [output] = calculateFpGrf(data,fp, rotm, grf_cutoff)
%calculateFpGrf Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    data
    fp
    rotm
    grf_cutoff
end

arguments (Output)
    output
end
corners = fp.corners;
origin = fp.origin;
ref_frame = computePfReferenceFrame(corners);
mean_corners = mean(corners,2);

length_frames = length(data);

force = data(:,1:3);
moment = data(:,4:6);

output = zeros(length_frames,9);
for i = 1:length_frames

    f = force(i,:);
    f_raw = f;
    m = moment(i,:);
    m_raw = m + cross(f,origin);

    fz = f(3);
    valid = -fz >= grf_cutoff;

    cop_raw = [-m_raw(2) / fz, m_raw(1) / fz, 0];

    f = ref_frame * f_raw';
    % m = ref_frame * m_raw';
    cop = ref_frame * cop_raw' + mean_corners;
    tz = ref_frame * (m_raw' - cross(f_raw', -1 .* cop_raw'));

    if ~valid
        cop = mean_corners;
        tz = ref_frame * (m_raw');
    end

    f = rotm * f;
    % m = rotm * m;
    cop = rotm * cop;
    tz = rotm * tz;

    % Convert from mm to m
    cop = cop ./ 1000;
    tz = tz ./ 1000;
    % m = m ./1000;

    % Corresponds with OpenSim extract ForceLocation::CenterOfPressure
    % Force 1:3
    % COP 4:6
    % Moment 7:9
    output(i,1:3) = f;
    output(i,4:6) = cop;
    output(i,7:9) = tz;
end
end