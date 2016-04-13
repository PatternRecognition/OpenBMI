function [out] = prep_meanInterval(dat,varargin)
% This function calculates the mean in specified time interval for each epoch
%
% Example:
% out = prep_meanInterval(dat,{'Time',[1000 2000]})
%         : 1 mean value for each epoch, calculated from 1s to 2s
% out = prep_meanInterval(dat,{'Time',[1000 2000];'Means',20})
%         : 20 mean values for each epoch, calculated from 1s to 2s
% out = prep_meanInterval(dat,{'Means',20})
%         : 20 mean values for each epoch
%
% Input:
%     dat - Epoched data structure
%     Time - Time interval you want to calculate the mean
%     Means - The number of means you want to calculate in a single epoch
%
% Returns:
%     out - Averaged data structure
% 
% 
% Seon Min Kim, 04-2016
% seonmin5055@gmail.com

if ~isfield(dat,'x')
    warning('OpenBMI: Data structure must have a field named ''x''');return
end
if ndims(dat.x)~=3
    warning('OpenBMI: Data must be segmented');return
end
if isempty(varargin)
    warning('OpenBMI: Whole samples in each trial are averaged')
    out = rmfield(dat,'x');
    out.x = mean(dat.x,1);
    if isfield(out,'fs')
        out = rmfield(out,'fs');end
    if isfield(out,'ival')
        out = rmfield(out,'ival');end
    return
end
opt = opt_cellToStruct(varargin{:});

if isfield(opt,'Time')
    if ~isnumeric(opt.Time)
        warning('OpenBMI: Time interval should be in a array');return
    end
    if ~isfield(dat,'ival')
        warning('OpenBMI: Data structure must have a field named ''ival''');return
    end
    dat.x = dat.x((dat.ival>=opt.Time(1) & dat.ival<=opt.Time(end)),:,:);
    if ~isfield(opt,'Means')
        x = mean(dat.x,1);
    elseif isfield(opt,'Means') && isscalar(opt.Means)
        out = prep_meanInterval(dat,{'Means',opt.Means});return
    else
        warning('OpenBMI: ''Means'' should be a scalar');return
    end
elseif isfield(opt,'Means') && isscalar(opt.Means)
    m = opt.Means;
    n = floor(size(dat.x,1)/m);
    x = zeros(m,size(dat.x,2),size(dat.x,3));
    for j=1:m
        x(j,:,:) = mean(dat.x([(j-1)*n+1:j*n],:,:),1);
    end
else
    warning('OpenBMI: ''Means'' should be a scalar');return
end

out = rmfield(dat,'x');
out.x = x;
out.meanOpt = opt;

if isfield(out,'fs')
    out = rmfield(out,'fs');end
if isfield(out,'ival')
    out = rmfield(out,'ival');end
