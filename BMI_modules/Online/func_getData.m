function [ fbData, opt ] = func_getData( eegFb, time, opt)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if ~isfield(eegFb, 'x')
    error('Test data should be field of eeg.x');
end

dat=eegFb.x(time.buffer{str2num(opt.ite)},:);
fbData=eegFb;
fbData.x=dat;
end

