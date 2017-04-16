function [ dat ] = prep_baselineCorrection( dat, varargin )
%PREP_BASELINECORRECTION Summary of this function goes here
%   Detailed explanation goes here
%% dat = .x, .fs
if ~varargin{end}
    varargin=varargin{1,1}; %cross-validation procedures
end;
if isempty(varargin)
    warning('Varargin parameter is not valid')
else
    ival=varargin{1};
end
if ~isfield(dat, 'x')
    warning('dat.x is not valid');
end

if ~isfield(dat, 'fs')
    warning('dat.fs is not valid');
end

fs=dat.fs;
[nDat nTrial nChan]=size(dat.x);
idc= floor(ival(1)*fs/1000)+1:ceil(ival(2)*fs/1000);
baseline=nanmean(dat.x(idc,:,:),1);
dat.x=dat.x- repmat(baseline, [nDat 1 1]);

% stack
if isfield(dat, 'stack')
    c = mfilename('fullpath');
    c = strsplit(c,'\');
    dat.stack{end+1}=c{end};
end
end

