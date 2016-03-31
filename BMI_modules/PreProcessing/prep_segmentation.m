function [ epo ] = prep_segmentation( dat, marker, varargin )
%PROC_EPOCHING Summary of this function goes here
%   Detailed explanation goes here
% if ~varargin{end}
%     varargin=varargin{1,1}; %cross-validation procedures
% end;

opt=opt_cellToStruct(varargin{:});
epo=struct('x',[],'t',[],'fs',[],'y',[],'y_logic',[],'chSet',[],'class',[]);
if ~isfield(opt,'fs')
    if isfield(dat,'fs')
        opt.fs=dat.fs;
        epo.fs=dat.fs;
    else
        error('Parameter is missing: fs');
    end
end

if isfield(dat,'x')
    tDat=dat.x;
    [nDat, nCh]=size(tDat);
else
    error('Parameter is missing: dat.x');
end

if isfield(marker,'y') && isfield(marker,'t')
    tMrk.y=marker.y
    tMrk.t=marker.t
    epo.y=marker.y;
else
    error('Parameter is missing: marker.y');
end

if isfield(marker,'y_logic')
    epo.y_logic=marker.y_logic
end

if isfield(marker,'class')
    epo.class=(marker.class);
end

if isfield(dat,'chSet')
    epo.chSet=dat.chSet;
end

fs=opt.fs;
ival=opt.interval;
idc= floor(ival(1)*fs/1000):ceil(ival(2)*fs/1000);
T= length(idc);
nEvents= size(tMrk.t, 2);
nChans= nCh;
% round
IV= round(idc(:)*ones(1,nEvents) + ones(T,1)*tMrk.t);
epo.x= reshape(tDat(IV, :), [T, nEvents, nChans]);
epo.t= linspace(ival(1), ival(2), length(idc));

% stack
% if isfield(eeg, 'stack')
%     c = mfilename('fullpath');
%     c = strsplit(c,'\');
%     epo.stack{end+1}=c{end};
% end

end

