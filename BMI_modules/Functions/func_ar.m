function [out] = func_ar(dat, order, varargin)
% func_ar (Feature extraction) :
%     This function calculates the autoregression(AR) parameter.
%
% Example:
%     [out] = func_ar(dat, 7, {'method','arburg'})
%
% Returns:
%     dat    - Data structure, segmented
%     order  - Order of AR setting
%
% Option: models for obtatining AR parameter
%     method - 'aryule'(default), 'arburg', 'arcov', 'armcov'

if nargin <= 1
    error('OpenBMI: Input data should be specified');
end

if ~all(isfield(dat, 'x'))
   error('OpenBMI: Data must have fields named ''x''');
end

opt = opt_cellToStruct(varargin{:});

if ~isfield(opt, 'method') %method selection
    opt.method = 'aryule';
end

[~, nEvents , nChans] = size(dat.x);
temp_ar = [];

for i = 1:nChans * nEvents
    ar = feval(opt.method, dat.x(:, i), order);
    temp_ar(:, i) = ar(2:end)';
end

dat.x = temp_ar;