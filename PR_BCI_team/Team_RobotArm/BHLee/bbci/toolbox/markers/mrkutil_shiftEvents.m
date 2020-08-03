function mrk = mrkutil_shiftEvents(mrk, offset, ix_events)
%% function mrk = mrkutil_shiftEvents(mrk, offset, classes)
% function to shift events in time. The offset is specified in ms, with
% positive offsets resulting in delayed later events. This function is
% useful to correct for latency/jitter in ERP eperiments. Here the actual 
% stimulus might be starting e.g. 50ms after the trigger (marker). 
% 
% INPUT
%   mrk       marker structure with .pos and .fs
%   offset    offset for the shift in time (in ms). May be scalar or vector (specifying an individual offset)
%   ix_events (OPTIONAL) indices of events that will be shifted, default 1:length(mrk.pos)
%   
% OUTPUT
%   mrk       updated marker structure
% 
%  JohannesHoehne 04/2012
  

if nargin < 3
    ix_events = 1:length(mrk.pos);
end

if length(offset) == 1
    offset = repmat(offset, 1, length(ix_events));
end


shift = round(offset / (1000 / mrk.fs));
if ~isequal(round(offset / (1000 / mrk.fs)), offset / (1000 / mrk.fs))
    warning('marker shifts coudl not be done precisely. Increase the sampling rate to get a precise shift')
end

mrk.pos(ix_events) = mrk.pos(ix_events) + shift;
