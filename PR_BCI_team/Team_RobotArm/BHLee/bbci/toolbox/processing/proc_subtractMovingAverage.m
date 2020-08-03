function dat= proc_subtractMovingAverage(dat, ms, varargin)
%dat= proc_subtractMovingAverage(dat, msec, <method='causal'>)
%
% IN   dat    - data structure of continuous or epoched data
%      msec   - length of interval in which the moving average is
%               to be calculated in msec
%      method - 'centered' or 'causal' (default)
%
% OUT  dat       - updated data structure

% bb, ida.first.fhg.de


nSamples= getIvalIndices(ms, dat.fs);
sx= size(dat.x);
xa= movingAverage(dat.x(:,:), nSamples, varargin{:});
xa= reshape(xa, sx);
dat.x= dat.x - xa;
