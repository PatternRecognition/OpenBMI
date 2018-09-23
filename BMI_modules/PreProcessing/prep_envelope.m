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
% Min-ho Lee, 09-2018
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isfield(dat, 'x')
    warning('OpenBMI: Data structure must have a field named ''x''')
    return
end
if ndims(dat.x) ~= 3
    warning('OpenBMI: Data must be segmented. Is the number of channel 1?');
end

s = size(dat.x);
dat.x= reshape(abs(hilbert(dat.x(:, :))),s);
out= prep_movingAverage(dat,varargin{:});
if ~exist('opt')
    opt = struct([]);
end
if ~isfield(dat, 'history')
    out.history = {'prep_envelope', opt};
else
    out.history(end + 1, :) = {'prep_envelope', opt};
end
%%--> 2016a 이상이면 prep_movingAverage 없앨수 있음…