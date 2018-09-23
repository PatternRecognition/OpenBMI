function [out] = prep_movingAverage(dat, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_MOVINGAVERAGE - averages the data with moving time window
% prep_movingAverage (Pre-processing procedure):
% 
% Synopsis:
%         [out] = prep_movingAverage(DAT, <OPT>)
%
% Example:
% [out] = prep_movingAverage(dat, {'time', 80})
% [out] = prep_movingAverage(dat, {'method', 'centered'})
% 
% Arguments:
%     dat - Structure. Continuous data or epoched data (data.x)
%     varargin - property/value list of optional properties
%
% Returns: Data structure which the function averages
%
% Options:
%     Time[ms] - time window. scalar or nx1 vector for weighting (default:100)
%     Method - 'centered' or 'causal' (default:causal)
%     Samples - the number of samples in a single time window. scalar
%               If the 'Time' option exists, this option will be ignored.
%
% Description:
% This function averages the data with moving time window
% 
% Seon Min Kim, 09-2018
% seonmin5055@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. window에 weighting하는 경우 고려 필요. window는 scalar나 vector
% 2. samples 수 고려. time이랑 같이 들어오면 무시
% 3. NaN이 포함될 경우.

if isempty(varargin)
    opt.time = 100;
    opt.method = 'centered';
else
    opt = opt_cellToStruct(varargin{:});
    if ~isfield(opt, 'time')
        opt.time = 100;
    end
    if ~isfield(opt, 'method')
        opt.method = 'causal';
    end
end

if ~isfield(dat, {'x', 'fs'})
    warning('OpenBMI: Data structure must have a field named ''x'' and ''fs''');
    return
end

if all(ndims(dat.x) ~= [2 3])
    warning('OpenBMI: Data dimension must be 2 or 3');
    return
end

% t = size(dat.x, 1);
% n = round(opt.Time*dat.fs/1000);
% switch opt.Method
%     case 'causal'
%         x = movmean(dat.x, [min(n,t)-1 0]);
%     case 'centered'
%         x = movmean(dat.x, opt.Time);
% end
% out = rmfield(dat, 'x');
% out.x = x;

if ndims(dat.x) == 3
    xx = zeros(size(dat.x));
    for i = 1:size(dat.x, 2)
        x = squeeze(dat.x(:, i, :));
        temp = rmfield(dat, 'x');
        temp.x = x;
        temp2 = prep_movingAverage(temp,varargin{:});
        xx(:, i, :) = temp2.x;
    end
    out = rmfield(dat,'x');
    out.x = xx;
    return
end

[t, ~] = size(dat.x);
n = round(opt.time * dat.fs / 1000);
x = zeros(size(dat.x));

switch opt.method
    case 'causal'
        for i = 1:min(n, t)
            x(i, :) = mean(dat.x([1:i], :));
        end
        for i = n + 1:t
            x(i, :) = mean(dat.x([i - n + 1:i], :));
        end
    case 'centered'
        ws = -floor(n / 2);
        we = ws + n - 1;
        for i = 1:-ws + 1
            x(i, :) = mean(dat.x([1:i + we], :));
        end
        for i = -ws + 2:t - we
            x(i, :) = mean(dat.x([i + ws:i + we], :));
        end
        for i = t - we + 1:t
            x(i, :) = mean(dat.x([i + ws:end], :));
        end
end

out = rmfield(dat, 'x');
out.x = x;
