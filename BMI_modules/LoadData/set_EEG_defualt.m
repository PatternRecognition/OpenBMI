function [ eeg ] = set_EEG_defualt(hdr, opt)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

eeg.x= zeros(hdr.NumberOfChannels,100000);

if ~isfield(hdr,'NumberOfChannels') || ~isfield(opt,'fs') 
    disp('Important parameter is missing, check the hdr.NumberOfChannels and opt.fs')
end
eeg.fs=opt.fs;
eeg.numberofchannel=hdr.NumberOfChannels;
eeg.chloc=[];
eeg.stack={};

end

