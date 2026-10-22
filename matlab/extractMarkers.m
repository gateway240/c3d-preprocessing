function [status, output] = extractMarkers(c3d, file, input_path, rotm, ...
     time_start, time_end)
%extractMarkers  Marker Extraction - TRIAL_markers.trc file
%   Detailed explanation goes here
arguments (Input)
    c3d
    file
    input_path
    rotm
    time_start
    time_end
end

arguments (Output)
    status
    output
end

point_frequency = c3d.parameters.POINT.RATE.DATA;
point_units = c3d.parameters.POINT.UNITS.DATA{1};
% Build header
point_columns = string(c3d.parameters.POINT.LABELS.DATA);
invalid = contains(point_columns, "*");
point_columns = point_columns(~invalid);
% Get data
points = c3d.data.points;
points = points(:, ~invalid, :);
% Linear Interpolation
for m = 1:size(points,2)
    for d = 1:3
        points(d,m,:) = fillmissing( ...
            squeeze(points(d,m,:)), ...
            'linear', ...
            'EndValues','nearest');
    end
end
[status, output] = writeMarkerTrcFile(points, point_frequency, point_units,...
    point_columns, input_path, file,  rotm, time_start, time_end);
end