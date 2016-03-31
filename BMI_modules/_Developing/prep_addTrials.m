function [data, marker] = prep_addTrials(data1, marker1, data2, marker2)
% prep_addTrials (Pre-processing procedure):
%
% This function add the latter data(data2) to the former data(data1) and
% their corresponding markers.
%
% Example:
% [data, marker] = prep_addTrials(data1, marker1, data2, marker2)
%
% Input: 
%     data1, marker1 - Data and marker structure, continuous or epoched
%     data2, marker2 - Data and marker structure to be added to data1
%
% Returns:
%     data - Updated data structure
%     marker - Updated marker structure
%
%
% Seon Min Kim, 03-2016
% seonmin5055@gmail.com


dim = ndims(data1.x);

switch dim
    case 2
        if ~isequal(size(data1.x,2),size(data2.x,2))
            error ('Unmatched the number of channels')
        end
    case 3
        if ~isequal(size(data1.x,1),size(data2.x,1))
            error('Unmatched data size')
        elseif ~isequal(size(data1.x,3),size(data2.x,3))
            error('Unmatched the number of channels')
        end
end

if ~isequal(marker1.class,marker2.class)
    error ('Unmatched class info.')
end

data = data1;
marker = marker1;

data.x = cat(dim-1, data1.x, data2.x);
marker.y = cat(2, marker1.y, marker2.y);
marker.t = cat(2, marker1.t, marker2.t);
marker.y_class = cat(2, marker1.y_class, marker2.y_class);
marker.y_logic = cat(2, marker1.y_logic, marker2.y_logic);

if isfield(data1.y) && isfield(data2.y)
    data.y = cat(2, data1.y, data2.y);
    data.y_logic = cat(2, data1.y_logic, data2.y_logic);
end