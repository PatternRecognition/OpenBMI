function [dat,state]= online_subtractMovingAverage(dat, state, ms,varargin)
%[dat,state]= state_subtractMovingAverage(dat,state, msec, <method='centered'>)
%
% IN   dat    - data structure of continuous or epoched data
%      msec   - length of interval in which the moving average is
%               to be calculated in msec
%      method - 'centered' or 'causal'
%
% OUT  dat       - updated data structure

% bb, ida.first.fhg.de

[dat2,state] = online_movingAverage(dat,state,ms,varargin{:});

dat.x = dat.x-dat2.x;

