function marker_out= bbci_apply_queryMarker(marker, ival, mrkDesc)
%BBCI_APPLY_QUERYMARKER - Check for acquired markers
%
%Synopsis:
%  MARKER= bbci_apply_queryMarker(MARKER, IVAL)
%  MARKER= bbci_apply_queryMarker(MARKER, IVAL, MARKER_DESC)
%  MARKER= bbci_apply_queryMarker(MARKER, LEN)
%  MARKER= bbci_apply_queryMarker(MARKER, LEN, MARKER_DESC)
%
%Arguments:
%  MARKER - Structure of recently acquired markers;
%           field of 'data' structure of bbci_apply, see bbci_apply_structures
%  IVAL - Time interval which is checked for markers [msec]
%      The time interval is relative to the start of bbci_apply.
%      ?? interval is 'open' on the left side ??
%  LEN - Length [msec] of the time interval that is to be checked
%        counted backwards from the last time point of acquisition, 
%        i.e., 100 relates to the most recent 100 msec of data.
%  MARKER_DESC - can either be a vector (format 'numeric') or a
%      string (or cell array of strings). It specifies the marker(s) which
%      is/are queried, e.g., [41 255] or {'S 41','S255'}.
%
%Output:
%  MARKER - Structure specifying all markers in the queried interval
%           fields 'time', 'desc'. If MARKER_DESC is specified, only
%           markers being members thereof are returned.

% 02-2011 Benjamin Blankertz


TIME_EPS= 0.001;

if length(ival)==1,
  ival= [-ival 0] + marker.current_time;
end

%idx= find(marker.time > ival(1) & marker.time<= ival(2));
idx= find(marker.time > ival(1)+TIME_EPS & marker.time<= ival(2)+TIME_EPS);

if nargin > 2,
  idx2= find(ismember(marker.desc(idx), mrkDesc));
  idx= idx(idx2);
end

if isempty(idx),
  marker_out= [];
else
  if iscell(marker.desc),
    marker_out= struct('time', num2cell(marker.time(idx)), ...
                       'desc', marker.desc(idx));
  else
    marker_out= struct('time', num2cell(marker.time(idx)), ...
                       'desc', num2cell(marker.desc(idx)));
  end
end
