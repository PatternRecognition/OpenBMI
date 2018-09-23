function [out] = prep_filter( dat, varargin )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_FILTER - filters the data within specified frequency band
% prep_filter (Pre-processing procedure):
%
% Synopsis:
%     [out] = prep_filter(DAT,<OPT>)
%
% Example:
%     out = prep_filter(dat, {'frequency', [7 13];'fs',100});
%     out = prep_filter(dat, {'frequency', [7 13]});
%     out = prep_filter(dat, [7 13]);
%
% Arguments:
%     dat       - EEG data structure
%     varargin - struct or property/value list of optional properties:
%          : frequency - Frequency range that you want to filter
%          : fs        - Sampling frequency
% Returns:
%     out       - Spectrally filtered data
% 
% Description:
%     This function filters the data within specified frequency band
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
% Min-ho Lee
% mhlee@image.korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin)
    warning('OpenBMI: Frequency band should be specified')
    return
end

if isnumeric(varargin{1})
    opt.frequency = varargin{1};
elseif iscell(varargin{:})
    opt=opt_cellToStruct(varargin{:});
elseif isstruct(varargin{:}) % already structure(x-validation)
    opt=varargin{:};
end

switch isstruct(dat)
    case true %struct        
        if ~isfield(dat, 'x') && ~isfield(dat, 'fs')
            warning('Parameter is missing: dat.x or dat.fs');
        end
        
        if ~isfield(opt,'fs')
            if isfield(dat, 'fs')
                opt.fs=dat.fs;
            else
                error('Parameter is missing: fs');
            end
        end
        tDat=dat.x;
        if ndims(tDat)==3   %smt
            [nD nT nC]=size(tDat);
            tDat=reshape(tDat, [nD*nT,nC]);
            band=opt.frequency;
            [b,a]= butter(5, band/(opt.fs/2),'bandpass');
            tDat(:,:)=filter(b, a, tDat(:,:));
            tDat=reshape(tDat, [nD, nT,nC]);
            dat.x=tDat;
        elseif ndims(tDat)==2  %cnt
            band=opt.frequency;
            [b,a]= butter(5, band/(opt.fs/2),'bandpass');
            tDat(:,:)=filter(b, a, tDat(:,:));
            dat.x=tDat;
        end
        
        
    case false
        % add if dat is not struct
        tDat=dat;        
        if ndims(tDat)==3   %smt
            [nD nT nC]=size(tDat);
            tDat=reshape(tDat, [nD*nT,nC]);
            band=opt.frequency;
            [b,a]= butter(5, band/(opt.fs/2),'bandpass');
            tDat(:,:)=filter(b, a, tDat(:,:));
            tDat=reshape(tDat, [nD, nT,nC]);
            dat.x=tDat;
        elseif ndims(tDat)==2  %cnt
            band=opt.frequency;
            [b,a]= butter(5, band/(opt.fs/2),'bandpass');
            %% TODO: It' doesn't work when the sampling rate is 1000 Hz;
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
out = dat;
if ~exist('opt','var')
    opt = struct([]);
end
if ~isfield(dat,'history')
    out.history = {'prep_filter',opt};
else
    out.history(end+1,:) = {'prep_filter',opt};
end