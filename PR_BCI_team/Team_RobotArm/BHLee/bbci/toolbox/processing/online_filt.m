function [dat,state]= online_filt(dat, state, b, a, varargin)
%[cnt,state]= online_filt(cnt, state, b, a, <b2, a2, ...>)
%
% apply digital (FIR or IIR) forward filter(s)
%
% IN   cnt    - data structure of continuous data
%      b, a   - filter coefficients
%
% OUT  cnt    - updated data structure
%
% SEE online_filterbank for applying multiple filters in parallel

% bb, ida.first.fhg.de


[dat.x, state] = filter(b, a, dat.x, state, 1);
