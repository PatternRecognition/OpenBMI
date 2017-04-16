function [ dat ] = prep_segmentation( dat, varargin )
%PROC_EPOCHING Summary of this function goes here
%   Detailed explanation goes here
% if ~varargin{end}
%     varargin=varargin{1,1}; %cross-validation procedures
% end;

if iscell(varargin{:})
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) % already structure(x-validation)
    opt=varargin{:}
end

if ~isfield(opt,'interval')
    error('OpenBMI: parameter "interval" is missing')
end

if isfield(dat,'x')
    tDat=dat.x;
    [nDat, nCh]=size(tDat);
else
    error('OpenBMI: parameter "dat.x" is missing');
end

if ~isfield(dat,'t')
    error('OpenBMI: parameter "dat.t" is missing');
end

if isfield(dat,'fs')
    fs=dat.fs;
else
    error('OpenBMI: parameter "dat.fs" is missing');
end

ival=opt.interval;
idc= floor(ival(1)*fs/1000):ceil(ival(2)*fs/1000);
T= length(idc);
nEvents= size(dat.t, 2);
nChans= nCh;
% round
IV= round(idc(:)*ones(1,nEvents) + ones(T,1)*dat.t);
dat.x= reshape(tDat(IV, :), [T, nEvents, nChans]);
dat.ival= linspace(ival(1), ival(2), length(idc));

% stack
% if isfield(eeg, 'stack')
%     c = mfilename('fullpath');
%     c = strsplit(c,'\');
%     epo.stack{end+1}=c{end};
% end

end

