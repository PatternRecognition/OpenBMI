function [ dat ] = prep_filter( dat, varargin )
%PROC_FILTER Summary of this function goes here
% EEG.data=prep_filter(EEG.data, {'frequency', [7 13]});
% data=prep_filter(data, {'frequency', [7 13];'fs',100 });
%   Detailed explanation goes here
% if ~varargin{end}
%     varargin=varargin{1,1}; %cross-validation procedures
% end;

if ~isfield(dat, 'x') && ~isfield(dat, 'fs')
    warning('Parameter is missing: dat.x or dat.fs');
end

opt=opt_cellToStruct(varargin{:});
if ~isfield(opt,'fs')
    if isfield(dat, 'fs')
        opt.fs=dat.fs;
    else
        error('Parameter is missing: fs');
    end
end
% if length(varargin)>1; param=opt_proplistToCell(varargin{:});end

switch isstruct(dat)
    case true %struct        
        tDat=dat.x;            
        band=opt.frequency;
        [b,a]= butter(5, band/opt.fs*2,'bandpass');
        tDat(:,:)=filter(b, a, tDat(:,:));  
        dat.x=tDat;
    case false
        % add if dat is not struct
        tDat=dat;            
        band=opt.frequency;
        [b,a]= butter(5, band/opt.fs*2,'bandpass');
        tDat(:,:)=filter(b, a, tDat(:,:));                
        fld='x';
        dat=tDat;       
end

% History
if isfield(dat,'stack')
    c = mfilename('fullpath');
    c = strsplit(c,'\');
    dat.stack{end+1}=c{end};
end
end



