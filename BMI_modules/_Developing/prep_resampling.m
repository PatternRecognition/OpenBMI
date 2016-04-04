function [data] = prep_resampling(data, fs, varargin)
% prep_resampling (Pre-processing procedure):
%
% Function will be revised later, including marker input and with corrected time variable
% 
% This function changes the sampling rate of the given EEG signal
% using a poly phase filter implementation. (up/downsampling)
%
% Example:
% [dat, marker] = prep_resampling(dat, marker, fs, {'Nr', n_samples})
%
% Input:
%     dat(struct) - Data structure to be changed, continuous or epoched
%     marker(struct) - Marker structure with field 't' (time)
%     fs(scalar, Hz) - Desired sampling frequency
%
%     varargin(optional):
%         Nr(scalar) - Remove the first and last Nr samples from the resampled data
%             to avoid edge effects (default is 0)
%
% Returns:
%     dat - Updated data structure
%     marker - Updated marker structure
%
%
% Seon Min Kim, 03-2016
% seonmin5055@gmail.com


if isempty(varargin)
    disp('The number of removed samples is 0 (default)');
    Nr = 0;
elseif ~iscell(varargin)
    error('myApp:argChk','Number of samples to be removed should be input in a cell type. \r\n For example: {''Nr'',10}')
else
    Nr = cell2mat(varargin{1}(2));   % should be revised
end

% For now, fs should be integer
% This part should be revised later
p = fs;
q = data.fs;

% Resamples the sequence at p/q times the original sampling rate
n_t = size(data.x,1);
n_tr = size(data.x,2);
n_ch = size(data.x,3);
x = zeros(ceil(n_t*p/q),n_tr,n_ch);    % should be revised (1dim size changes after resampling)
if length(size(data.x)) == 3        % For epoched data (time x trial x channel)
    for i = 1:n_tr
        xt = reshape(data.x(:,i,:),[n_t,n_ch]);
        xt = resample(xt,p,q);
        x(:,i,:) = reshape(xt,[size(xt,1),1,n_ch]);
    end
else        % For continuous data, not epoched (time x channel)
    x = resample(data.x,p,q);
end

data.x = x;
data.fs = fs;
% Remove the N samples at both beginning and the end, to avoid edge effects
data.x = data.x((Nr+1):end-Nr,:,:);
