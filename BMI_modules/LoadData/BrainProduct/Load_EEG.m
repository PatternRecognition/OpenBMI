function [ dat, marker, hdr] = Load_EEG( file, varargin )
%LOAD_ Summary of this function goes here
%   Detailed explanation goes here

% opt=opt_proplistToStruct(varargin{:});
if ~isempty(varargin)
    opt=opt_cellToStruct(varargin{:});
else % set default parameters here
    opt.device='brainVision';
    opt.fs=100;
end
switch lower(opt.device)
    case 'brainvision'
        hdr=Load_BV_hdr(file);disp('Loading EEG header file..');
        dat=Load_BV_data(file, hdr, opt);disp('Loading EEG data..');
        marker=Load_BV_mrk(file, hdr, opt);disp('Loading Marker file..');
        dat.chSet=hdr.chan;
    case 'emotive'
        
end
disp('Data loaded!');
end

