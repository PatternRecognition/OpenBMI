function [out] = prep_cutDim(dat)
% prep_cutDim (Pre-processing procedure):
%
% This function reduces dimensionality of the data, reducing all dimensions
% except for the trial dimension.
%
% Input:
%     dat - Segmented data structure, or the data matrix itself
%     (Result will be same as input when the data is continuous)
%
% Returns:
%     out - Stacked data with reduced dimensionality
%
%
% Seon Min Kim, 04-2016
% seonmin5055@gmail.com

if isstruct(dat)
    if ~isfield(dat, 'x')
        warning('OpenBMI: Data structure must have a field named ''x''');
        return
    end
    x = dat.x;
elseif isnumeric(dat)
    x = dat;
else
    warning('OpenBMI: Check the type of data')
    return
end

s = size(x);
if length(s) == 3
    x = permute(x,[1 3 2]);
end
x = reshape(x,[prod(s(1:end-1)),s(end)]);

if isstruct(dat)
    out = rmfield(dat,'x');
    out.x = x;
elseif isnumeric(dat)
    out = x;
end
