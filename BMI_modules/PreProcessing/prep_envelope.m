function [out] = prep_envelope(dat, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_ENVELOPE - smoothly outlines the extremes of an oscillating signal, continuous or epoched
% prep_envelope (Pre-processing procedure):
%
% Synopsis:
%   [out] = prep_envelope(DAT, <OPT>)
%
% Example :
%    [out] = prep_envelope(dat)
%     [out] = prep_movingAverage(dat, {'time', 80})
%    [out] = prep_envelope(dat, {'time', 100; 'method', 'casual'})
%
% Arguments:
%     dat    - Epoched signal
% Options:
%     time[ms] - time window. scalar or nx1 vector for weighting (default:100)
%     method - 'centered' or 'casual' (default:casual)
%
% Returns:
%     out - Envelope of the signal
%
% Description:
%     This function smoothly outlines the extremes of an oscillating
%     signal, continuous or epoched.
%     continuous data should be [time * channels]
%     epoched data should be [time * trials * channels]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Hong-Kyung kim, 09-2018
% hk_kim@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isfield(dat, {'x', 'fs'})
    warning('OpenBMI: Data structure must have a field named ''x'' and ''fs''');
    return
end
if ndims(dat.x) ~= 3
    warning('OpenBMI: Data must be segmented. Is the number of channel 1?');
end

if isempty(varargin)
    opt = struct();
elseif isnumeric(varargin{1})
    opt.time = varargin{1};
elseif isstruct(varargin{1})
    opt = varargin{:};
else
    opt = opt_cellToStruct(varargin{:});    
end

def_opt = struct('time', 100, 'method', 'casual');
opt = opt_defaultParsing(def_opt, opt);

t = size(dat.x);
dat.x = reshape(abs(hilbert(dat.x(:, :))), t);

n = round(opt.time*dat.fs/1000);
switch opt.method
    case 'casual'
        x = movmean(dat.x, [min(n,t(1))-1 0]);
    case 'centered'
        x = movmean(dat.x, n);
end
out = rmfield(dat, 'x');
out.x = x;

out = opt_history(out, mfilename, opt);
%%--> 2016a 이상이면 prep_movingAverage 없앨수 있음…