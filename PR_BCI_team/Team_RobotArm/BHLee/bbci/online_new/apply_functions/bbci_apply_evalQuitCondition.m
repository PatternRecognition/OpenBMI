function run= bbci_apply_evalQuitCondition(marker, bbci, log_fid)
%BBCI_APPLY_EVALQUITCONDITION - Evalutate whether bbci_apply should stop
%
%Synopsis:
%  RUN= bbci_apply_evalQuitCondition(MARKER, BBCI, <LOG_FID>)
%
%Arguments:
%  MARKER - Structure of recently acquired markers;
%           field of 'data' structure of bbci_apply, see bbci_apply_structures
%  BBCI - Structure of bbci_apply which specifies processing and classificaiton
%  LOG_FID - File identifier of log fil(s)
%
%Output:
%  RUN - BOOLEAN, flag which says whether bbci_apply should continue to run

% 02-2011 Benjamin Blankertz


persistent last_call

if nargin<3,
  log_fid= [];
end

if isempty(last_call),
  last_call= 0;
end
this_call= marker.current_time;
ival= [last_call this_call];
quit_markers= bbci_apply_queryMarker(marker, ival, bbci.quit_condition.marker);
quit_cond1= ~isempty(quit_markers);
if quit_cond1,
  str= '#Quit marker [%s] received: stopping.';
  bbci_log_write(log_fid, str, vec2str(quit_markers.desc));
end
last_call= this_call;

quit_cond2= marker.current_time/1000 >= bbci.quit_condition.running_time;
if quit_cond2,
  str= '#Specified running time of %ds reached: stopping.';
  bbci_log_write(log_fid, str, bbci.quit_condition.running_time);
end

run= ~quit_cond1 && ~quit_cond2;
