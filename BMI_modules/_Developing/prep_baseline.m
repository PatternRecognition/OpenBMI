function [out] = prep_baseline(dat,varargin)
% prep_baseline (Pre-processing procedure):
% 
% This function corrects the baseline by subtracting average amplitude in
% the specified interval from a segmented signal.
%
% Example:
% [out] = prep_baseline(dat,{'Time',[-100 0];'Criterion','class'})
%
% Input:
%     dat       - segmented data structure
% Option:
%     Time      - time interval. [start ms, end ms] or time(ms) from the
%                 beginning (default: all)
%     Criterion - 'class', 'trial', 'channel' (default: 'trial')
%
% Returns:
%     dat - baseline corrected data structure
%
%
% Seon Min Kim, 04-2016
% seonmin5055@gmail.com

if ~isfield(dat,'x')
    warning('OpenBMI: Data must have fields named ''x''');return
elseif ndims(dat.x)~=3
    warning('OpenBMI: Data must be segmented');return
elseif ~isfield(dat,'ival')
    warning('OpenBMI: Data must have fields named ''ival''');return
elseif ~isfield(dat,'fs')
    warning('OpenBMI: Data must have fields named ''fs''');return
end

if isempty(varargin)
    opt.Time = [dat.ival(1),dat.ival(end)];
    opt.Criterion = 'trial';
else
    opt = opt_cellToStruct(varargin{:});
end
if ~isfield(opt,'Time')
    opt.Time = [dat.ival(1),dat.ival(end)];
elseif ~isfield(opt,'Criterion')
    opt.Criterion = 'trial';
    if isscalar(opt.Time)
        opt.Time = [dat.ival(1),dat.ival(1)+opt.Time];
    elseif ~isvector(opt.Time)
        warning('OpenBMI: Time should be a scalar or a vector');return
    end
end
% Time interval이 ival보다 클 경우

[nT,~,~] = size(dat.x);
switch opt.Criterion
    case 'trial'
        t = opt.Time-dat.ival(1)+1;
        idx = (floor(t(1)*dat.fs/1000)+1):ceil(t(end)*dat.fs/1000);
        base = nanmean(dat.x(idx,:,:),1);
        x = dat.x-repmat(base,[nT,1,1]);
%     case 'class'
%         base = 
%     case 'channel'
%         base = 
end

out = rmfield(dat,'x');
out.x = x;
