function [out] = prep_erpMeans(dat,varargin)
% This function calculates the mean in specified time interval for each epoch
%
% Example:
% out = prep_erpMeans(dat,{'Time',[1000 2000]})
%         : 1 mean value for each epoch, calculated from 1s to 2s
% out = prep_erpMeans(dat,{'Means',20})
%         : 20 mean values for each epoch
% out = prep_erpMeans(dat,{'Samples',50})
%         : calculate means with 50 samples each
% Two or three options can be used together, but when the both 'Means' and
% 'Samples' are used,(e.g.{'Time',[1000 2000];'Means',20}) 'Means' will be ignored.
%
% Input:
%     dat - Epoched data structure
% 
% Options:
%     Time - Time interval you want to calculate the mean
%     Means - The number of means you want to calculate in a single epoch
%     Samples - The number of samples used in calculating a single mean value
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
    t_iv = (dat.ival>=opt.Time(1) & dat.ival<=opt.Time(end));
    dat.x = dat.x(t_iv,:,:);
    dat.ival = dat.ival(t_iv);
    if ~isfield(opt,'Samples') && ~isfield(opt,'Means')
        opt.Samples = size(dat.x,1);opt.Means = 1;
        x = mean(dat.x,1);
    elseif isfield(opt,'Samples') && isscalar(opt.Samples)
        if isfield(opt,'Means')
            warning('OpenBMI: Option for ''Means'' will be ignored')
        end
        out = prep_erpMeans(dat,{'Samples',opt.Samples});return
    elseif ~isfield(opt,'Samples') && isfield(opt,'Means') && isscalar(opt.Means)
        out = prep_erpMeans(dat,{'Means',opt.Means});return
    else
        warning('OpenBMI: Options for ''Samples'' and/or ''Means'' should be a scalar');return
    end
else
    if isfield(opt,'Means') && isscalar(opt.Means)
        m = opt.Means;
        s = floor(size(dat.x,1)/m);
        opt.Samples = s;
    elseif isfield(opt,'Samples') && isscalar(opt.Samples)
        s = opt.Samples;
        m = floor(size(dat.x,1)/s);
        opt.Means = m;
    else
        warning('OpenBMI: ''Means'' should be a scalar');return
    end
    x = zeros(m,size(dat.x,2),size(dat.x,3));
    for j=1:m
        x(j,:,:) = mean(dat.x([(j-1)*s+1:j*s],:,:),1);
    end
end

out = rmfield(dat,'x');
out.x = x;
out.meanOpt = opt;

if isfield(out,'fs')
    out.fs = out.fs/opt.Samples;end
if isfield(out,'ival')
    out.ival = mean(dat.ival(reshape(1:opt.Means*opt.Samples,opt.Samples,opt.Means)));end
