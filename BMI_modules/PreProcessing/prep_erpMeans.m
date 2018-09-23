function [out] = prep_erpMeans(dat, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_ERPMEANS - calculate the mean in specified time interval for each epoch
% prep_erpMeans (Pre-processing procedure):
%
% Synopsis:
%   [out] = prep_erpMeans(dat, <var>)
%
% Example :
%         out = prep_erpMeans(dat, 20)
%         out = prep_erpMeans(dat, {'n_means', 20})
%                 : 20 mean values for each epoch
%         out = prep_erpMeans(dat, {'n_samples', 50})
%                 : calculate means with 50 samples each
%         Two or three options can be used together, but when the both 'n_means' and
%         'n_samples' are used, 'n_means' will be ignored.
%
% Arguments:
%     dat - Segmented data itself
% Options:
%     n_means - The number of means you want to calculate in a single epoch
%     n_samples - The number of samples used in calculating a single mean value
%
% Returns:
%     out - Mean values of eeg signal
%
% Description:
%     This function calculates the mean in specified time interval for each
%     epoch.
%     continuous data should be [time * channels]
%     epoched data should be [time * trials * channels]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 01-2018
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin)
    warning('OpenBMI: Whole samples in each trial are averaged')
    out = mean(dat, 1);
    return
end
if isnumeric(varargin{1})
    opt.n_means = varargin{1};
else
    opt = opt_cellToStruct(varargin{:});
end

if ndims(dat) ~= 3
    warning('OpenBMI: Data must be segmented. Is the number of channel 1?');
end

if isfield(opt,'n_means') && isscalar(opt.n_means)
    m = opt.n_means;
    s = floor(size(dat,1) / m);
elseif isfield(opt,'n_samples') && isscalar(opt.n_samples)
    s = opt.n_samples;
    m = floor(size(dat,1) / s);
else
    warning('OpenBMI: ''n_means'' and ''n_samples'' should be a scalar');
    return
end

x = zeros(m, size(dat, 2), size(dat, 3));
for j = 1:m
    x(j, :, :) = mean(dat([1:s] + s * (j - 1), :, :));
end

% if ~exist('opt')
%     opt = struct([]);
% end
% if ~isfield(dat,'history')
%     out.history = {'prep_erpMeans',opt};
% else
%     out.history(end+1,:) = {'prep_erpMeans',opt};
% end

out = x;