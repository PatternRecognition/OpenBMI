function [ eeg, mrk, hdr] = Load_EEG_data( file, varargin )
%LOAD_ Summary of this function goes here
%   Detailed explanation goes here

opt=opt_proplistToStruct(varargin{:});

switch lower(opt.device)
    case 'brainvision'
        hdr=Load_BV_hdr(file);disp('Loading EEG header file..');
        eeg=Load_BV_data(file, hdr, opt);disp('Loading EEG data..');
        mrk=Load_BV_mrk(file, hdr, opt);disp('Loading Marker file..');
end
disp('Data loaded!');
end

