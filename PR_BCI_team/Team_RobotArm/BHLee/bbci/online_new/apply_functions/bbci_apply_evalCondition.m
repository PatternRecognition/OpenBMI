function events= bbci_apply_evalCondition(marker, data_control, bbci_control)
%BBCI_APPLY_EVALCONDITION - Evaluate conditions which trigger control signals
%
%Synopsis:
%  EVENTS= bbci_apply_evalCondition(MARKER, DATA_CONTROL, BBCI_CONTROL)
%
%Arguments:
%  MARKER - Structure of recently acquired markers
%  DATA_CONTROL - Structure holding the control signal and status
%      information on determining the control signal,
%      subfield of 'data' structure of bbci_apply
%  BBCI_CONTROL - Structure specifying the calculation of the control signal.
%      subfield of 'bbci' structure of bbci_apply. 
%
%Output:
%  EVENTS - Array of Struct, specifying the event(s) at which a control
%      signal should be calculated. Each Struct has the fields
%      'time' and 'desc' like in the marker context.

% 02-2011 Benjamin Blankertz


events= [];
bcc= bbci_control.condition;

% 1. Variant: determine control for most recent data segment
if isempty(bcc),
  events= struct('time', data_control.time, 'desc',[]);
  return;
end

% 2. Variant: determine control depending on the occurence of markers
if ~isempty(bcc.marker),
  check_ival= [data_control.lastcheck marker.current_time] - bcc.overrun;
  events= bbci_apply_queryMarker(marker, check_ival, bcc.marker);
  return;
end

% 3. Variant: determine control at a regular time interval, e.g. every 100msec
tim= data_control.lastcheck + bcc.interval;
if tim <= data_control.time,
  events= struct('time',tim, 'desc',[]);
end
