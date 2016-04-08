function [out] = prep_resample(dat, fs, varargin)
% prep_resampling (Pre-processing procedure):
% ***
% Changes in t and ival should be considered, resulted from edge effect
% 
% This function changes the sampling rate of the given EEG signal.
% It can do both up/downsampling, considering frequency up to 3 digits.
%
% Example:
% [out] = prep_resample(dat, fs, {'Nr', n_samples})
%
% Input:
%     dat    - dat structure to be changed, continuous or epoched
%     fs[Hz] - Desired sampling frequency (scalar)
%
% Options:
%     Nr(scalar) - Remove the first and last Nr samples from the
%                  resampled dat to avoid edge effects (default is 0)
%
% Returns:
%     dat - Updated dat structure
%
%
% Seon Min Kim, 03-2016
% seonmin5055@gmail.com


if isempty(varargin)
    disp('The number of removed samples is 0 (default)');
    opt.Nr = 0;
elseif ~iscell(varargin)
    warning('OpenBMI: Number of samples to be removed should be in a correct form, cell type')
    return
else
    opt = opt_cellToStruct(varargin{:});
end

if ~isfield(dat,'x') || ~isfield(dat,'fs')
    warning('OpenBMI: Data must have fields named ''x'',''fs''')
    return
end

p = round(1000*fs);
q = round(1000*dat.fs);

if ndims(dat.x) == 3
    n_t = size(dat.x,1);
    n_tr = size(dat.x,2);
    n_ch = size(dat.x,3);
    x = zeros(ceil(n_t*p/q),n_tr,n_ch);
    for i = 1:n_tr
        xt = reshape(dat.x(:,i,:),[n_t,n_ch]);
        xt = resample(xt,p,q);
        x(:,i,:) = reshape(xt,[size(xt,1),1,n_ch]);
    end
    x = x((opt.Nr+1):end-opt.Nr,:,:);
elseif ismatrix(dat.x)
    x = resample(dat.x,p,q);
    x = x((opt.Nr+1):end-opt.Nr,:);
else
    warning('Check for the dimension of input data')
    return
end

out = rmfield(dat,{'x','fs'});
out.fs = fs;
out.x = x;

if isfield(dat,'t')
    out.t = dat.t/dat.fs*fs;
end
if isfield(dat,'ival')
    out.ival = linspace(dat.ival(1),dat.ival(end),size(x,1));
end
