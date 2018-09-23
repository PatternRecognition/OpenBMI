function [out] = prep_selectTime_(dat, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_SELECTTIME - selects the part of a specific time interval from continuous or epoched data.
% prep_selectTime (Pre-processing procedure):
%
% Synopsis:
%     [out] = prep_selectTime(DAT, <OPT>)
%
% Example:
%     out = prep_selectTime(dat, {'time', [1000 3000]})
%     out = prep_selectTime(dat, [1000 3000])
%
% Arguments:
%     dat - Structure. epoched data
%     varargin - struct or property/value list of optional properties:
%           time: Time interval to be selected (ms)
%
% Returns:
%     out - Data structure which has selected time from epoched data
%
% Description:
%     This function selects the part of a specific time interval
%     from continuous or epoched data.
%     (i)  For continuous data, this function selects data in specifie time
%          interval from the whole data.
%     (ii) For epoched data, this function selects time interval in each trial.
%          If you want to select trials in specific time interval, you can use
%          a function 'prep_selectTrials'
%     continuous data should be [time * channels]
%     epoched data should be [time * trials * channels]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 12-2017
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

out = dat;

if isempty(varargin)
    warning('OpenBMI: Time interval should be specified')
    return
end

if iscell(varargin{1}) && strcmpi(varargin{1}{1}, {'time'})
    opt.time = varargin{1}{2};
elseif ismatrix(varargin{1})
    opt.time = varargin{1};
end

if ~isfield(opt, 'time')
    warning('OpenBMI: Time interval should be specified.')
    return
end

if ~all(isfield(dat, {'x', 't', 'chan', 'fs'}))
    warning('OpenBMI: Data must have fields named ''x'', ''t'', ''chan'', and ''fs''')
    return
end

ival = opt.time;
d = ndims(dat.x);
ival_start = ceil(ival(1) * dat.fs / 1000);
ival_end = floor(ival(2) * dat.fs / 1000);

if d == 3 || (d == 2 && length(dat.chan) == 1)
    if ~isfield(dat, 'ival')
        warning('OpenBMI: Epoched data must have fields named ''ival''')
    return
    end
    if ival(1) < dat.ival(1) || ival(2) > dat.ival(end)
        warning('OpenBMI: Selected time interval is out of epoched interval')
        return
    end
    iv = ival_start:ival_end - dat.ival(1) * dat.fs / 1000 + 1;
    x = dat.x(iv, :, :);
    if isfield(out, 'ival')
        out.ival = dat.ival(iv);
    end
elseif d == 2 && length(dat.chan) > 1
    if (ival(1) < 0) || ((ival(2) / 1000) > (size(dat.x, 1) / dat.fs))
        warning('OpenBMI: Selected time interval is out of time range')
        return
    end
    x = dat.x(ival_start:ival_end, :);
    s = find((dat.t * dat.fs / 1000) <= ival_start);
    e = find((dat.t * dat.fs / 1000) <= ival_end);
    if isempty(s) || isempty(e)
        warning('OpenBMI: Selected time interval is out of time range')
        return
    end
    iv = s(1):e(end);
    t = dat.t(iv);
    out.t = t;
end

out.x = x;

out = opt_history(out, 'prep_selectTime', opt);

end