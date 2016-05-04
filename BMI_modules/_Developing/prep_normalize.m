function [out] = prep_normalize(dat,varargin)
% prep_normalize (Pre-processing procedure):
%
% This function normalizes
%
% Example:
% [out] = prep_normalize(dat,{'Method','std'});
%
% Input:
%     dat - Feature vector, struct or data itself
% Options:
%     Method - 'std'
% 
% Seon Min Kim, 05-2016
% seonmin5055@gmail.com

% Options of normalizing methods should be added later

if isempty(varargin)
    opt.Method = 'std';
else
    opt = opt_cellToStruct(varargin{:});
end

if isstruct(dat)
    if ~isfield(dat,'x')
        warning('OpenBMI: Data structure must have a field named ''x''');return
    end
    fv=dat.x;
elseif isnumeric(dat)
    fv=dat;
else
    warning('OpenBMI: Check for format of the data');return
end
z = size(fv);
if length(z)~=2
    warning('OpenBMI: Data should be in a form of "feature x trial"');return
end

switch opt.Method
    case 'std'
        c = repmat(std(fv,0,2),[1,z(2)]);
        fv = fv.*(1./c);
    otherwise
        warning('OpenBMI: Unknown nomalizing method. It might be updated soon')
end

if isstruct(dat)
    out = rmfield(dat,'x');
    out.x = fv;
elseif isnumeric(dat)
    out = fv;
end
