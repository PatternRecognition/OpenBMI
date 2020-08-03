function dat= proc_movingAverage(dat, ms, varargin)
%PROC_MOVINGAVERAGE - Moving average (low-pass) filter
%
%Usage:
% DAT= proc_movingAverage(DAT, MSEC, <METHOD='causal'>)
%
%Input:
% DAT    - data structure of continuous or epoched data
% MSEC   - length of interval in which the moving average is
%          to be calculated, unit [msec].
% METHOD - 'centered' or 'causal' (default).
%
%Output:
% DAT    - updated data structure

% Author(s): Benjamin Blankertz


nSamples = round(ms*dat.fs/1000);
dat.x(:,:)= movingAverage(dat.x(:,:), nSamples, varargin{:});
