function [out] = prep_average(dat)
% prep_average (Pre-processing procedure):
%
% This function average the data for each class
%
% Example:
% [out] = prep_average(dat)
%
% Input:
%     dat - Epoched data structure
%
% Returns:
%     out - Averaged data structure, classwise
%
%
% Seon Min Kim, 04-2016
% seonmin5055@gmail.com

if ~isfield(dat,'x')
    warning('OpenBMI: Data structure must have a field named ''x''')
    return
end
if ~isfield(dat,'y_dec') || ~isfield(dat,'y_logic') || ~isfield(dat,'y_class')
    warning('OpenBMI: Class information is missing')
    return
end

if ndims(dat.x)~=3
    warning('OpenBMI: Data must be epoched')
    return
end

n_cls = size(dat.y_logic,1);
tri_cls = cell(1,n_cls);
for cls = 1:n_cls
    tri_cls{cls} = mean(dat.x(:,(dat.y_dec==cls),:),2);
end
x = cat(2,tri_cls{1:end});

out = rmfield(dat,{'t','y_dec','y_logic','y_class'});
out.x = x;
