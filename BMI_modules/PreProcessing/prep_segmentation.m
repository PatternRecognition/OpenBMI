function [out] = prep_segmentation(dat, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_SEGMENTATION - segment the data in a specific time interval based on the marked point
% prep_segmentation (Pre-processing procedure):
%
% Synopsis:
%     [out] = prep_segmentation(DAT, <OPT>)
%
% Example:
%     SMT=prep_segmentation(CNT, [750 3500])
%     SMT=prep_segmentation(CNT, {'interval', [750 3500]})
%
% Arguments:
%     dat - continuous EEG data structure
%     varargin - struct or property/value list of optional properties:
%           interval: time interval
%
% Returns:
%     out - segmented EEG data structure
%
% Description:
%     This function segments the data in a specific time interval based on
%     the marked point.
%     continuous data should be [time * channels]
%     epoched data should be [time * trials * channels]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 09-2018
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

out = dat;

if isempty(varargin)
    error('OpenBMI: Selected interval should be specified');
end

if isnumeric(varargin{1})
    opt.interval = varargin{1};
else
    opt = opt_cellToStruct(varargin{:});
end

if ~all(isfield(dat, {'x', 't', 'fs'}))
    error('OpenBMI: Data must have a field named ''x'', ''t'', and ''fs''')
end

% 수정하긴해야하는데...나중에 하자
ival = opt.interval;

tmp = dat.t;
tmp(1) = [];
% if any([ival(1) < 0, ival(2) < 0, ival(1) * dat.fs / 1000 > min(tmp - dat.t(1:end - 1)), ...
%         ival(2) * dat.fs / 1000 > min(tmp - dat.t(1:end - 1)), ival(2) < ival(1)])
%     ival = [0, min(tmp - dat.t(1:end - 1)) * 1000 / dat.fs];
%     warning('OpenBMI: Interval should be proper value, so we changed it. Please check.')
% end

idc = floor(ival(1) * dat.fs / 1000):ceil(ival(2) * dat.fs / 1000);
n_time = length(idc);
n_events = size(dat.t, 2);
n_chans = size(dat.x, 2);
% round
IV = round(idc(:) * ones(1, n_events) + ones(n_time, 1) * dat.t);
out.x = reshape(dat.x(IV, :), [n_time, n_events, n_chans]);
out.ival = linspace(ival(1), ival(2), length(idc));

out = opt_history(out, 'prep_segmentation', opt);

end