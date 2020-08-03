function epo= bbci_apply_getSegment(signal, reference_time, ival)
%BBCI_APPLY_GETSEGMENT - Retrieve segment of signals from buffer
%
%Synopsis:
%  EPO= bbci_apply_getSegment(SIGNAL, IVAL)
%
%Arguments:
%  SIGNAL - Structure buffering the continuous signals,
%      subfield of 'data' structure of bbci_apply
%  REFERENCE_TIME - Specifies the t=0 time point to which IVAL refers.
%      This is typically either the most recent time point (for continuous
%      classifcation), or the time point of an event marker.
%  IVAL - Time interval [start_msec end_msec] relative to the
%      REFERENCE_TIME that defines the segment within the continuous data.
%
%Output:
%  EPO - Structure of epoched signals with the fields
%        'x' (data matrix [time x channels]), 'clab', and 't' (time line).

% 02-2011 Benjamin Blankertz


% Determine the indices in the ring buffer that correspond to the specified
% time interval. There are some rounding-issues here for some sampling rates.
% The following procedure should do well.
len= floor(diff(ival)/1000*signal.fs);
idx0= signal.ptr + (reference_time-signal.time)*signal.fs/1000;
idx_ival= round(idx0 + ival(1)/1000*signal.fs) + [0:len];
idx= 1 + mod(idx_ival-1, signal.size);

% Get requested segment from the ring buffer and store it into an EPO struct
epo.x= signal.x(idx,:);
epo.clab= signal.clab;
timeival= (idx_ival([1 end])-idx0)*1000/signal.fs;
timeival= round(10000*timeival)/10000;
epo.t= linspace(timeival(1), timeival(2), length(idx));
epo.fs= signal.fs;
