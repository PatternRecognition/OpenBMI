function [out] = prep_baseline(dat,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_BASELINE - corrects the baseline by subtracting average amplitude in the specified interval from a segmented signal
% prep_baseline (Pre-processing procedure):
%
% Synopsis:
%   [out] = prep_baseline(dat,<var>)
%
% Example :
%   [out] = prep_baseline(dat,[-100 0])
%   [out] = prep_baseline(dat,{'Time',[-100 0];'Criterion','class'})
%
% Arguments:
%     dat - segmented data structure
%   Option:
%     Time      - time interval. [start ms, end ms] or time(ms) from the
%                 beginning (default: all)
%     Criterion - 'class', 'trial'(default: 'trial')
%
% Returns:
%     dat - baseline corrected data structure
%
%
% Description:
%     This function corrects the baseline by subtracting average amplitude
%     in the specified interval from a segmented signal.
%     continuous data should be [time * channels]
%     epoched data should be [time * trials * channels]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 12-2017
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out = dat;

if ~isfield(dat, {'x', 'ival', 'fs'})
    warning('OpenBMI: Data must have fields named ''x'', ''ival'', and ''fs''');
    return
end

if ndims(dat.x)~=3 && size(dat.chan,2)~=1
    warning('OpenBMI: Data must be segmented');
end

opt = opt_cellToStruct(varargin{:});
def_opt = struct('time', [dat.ival(1), dat.ival(end)], 'criterion', 'trial');
opt = opt_defaultParsing(def_opt, opt);

if isscalar(opt.time)
    opt.time = [dat.ival(1),dat.ival(1)+opt.time];
elseif ~isvector(opt.time)
    warning('OpenBMI: Time should be a scalar or a vector');return
end

t = opt.time-dat.ival(1)+1;

if t(1)<1 || t(end)>dat.ival(end)
    warning('OpenBMI: Selected time interval is out of time range');return
end

t_idx = floor(t(1)*dat.fs/1000+1):ceil(t(end)*dat.fs/1000);

switch opt.criterion
    case 'trial'
        base = nanmean(dat.x(t_idx,:,:));
        x = dat.x-base;
    case 'class'
        if ~isfield(dat,'y_logic')
            warning('OpenBMI: Data must have fields named ''y_logic''');return
        end
        x = zeros(size(dat.x));
        for i=1:size(dat.y_logic,1)
            cls_idx = dat.y_logic(i,:);
            base = nanmean(nanmean(dat.x(t_idx,cls_idx,:)),2);
            x(:,cls_idx,:) = dat.x(:,dat.cls_idx,:)-base;
        end
%     case 'channel'
%         base = 
end

out = rmfield(dat,'x');
out.x = x;

out = opt_history(out, mfilename, opt);
end