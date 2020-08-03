function mr = getActivationAreas(mrk, session_start_marker, session_end_marker, pause_start_marker, pause_end_marker)
%GETACTIVATIONAREA gives areas regarding mrk.pos back where the subject is not in a break
%
% usage:
%    area = getActivationArea(mrk,<session_start_marker, session_end_marker,pause_start_marker, pause_end_marker>);
%
% input:
%    mrk     a usual marker structure or a filename to load with readMarkerTable
%    session_start_marker:  an array of session_start_markers (252)
%    session_end_marker:  an array of session_end_markers (253)
%    pause_start_marker:  an array of pause_start_markers (249)
%    pause_end_marker:  an array of pause_end_markers     (250)
%
% output:
%   mr      an nx2 array with intervals
%
% GUIDO DORNHEGE; 19/03/2004

if ~exist('session_start_marker','var'),
  session_start_marker = 252;
end
if ~exist('session_end_marker','var'),
  session_end_marker = 253;
end
if ~exist('pause_start_marker','var'),
  pause_start_marker = 249;
end
if ~exist('pause_end_marker','var'),
  pause_end_marker = 250;
end


if ~isstruct(mrk)
  mrk = readMarkerTable(mrk);
end

classDef = {session_start_marker, session_end_marker, pause_start_marker, pause_end_marker;'session start','session end','pause start','pause end'};

mrk = makeClassMarkers(mrk,classDef,0,0);

mr = [];
status = 0;

for i = 1:size(mrk.y,2);
  switch status
   case 0   % no session started, wait for marker 1
    if mrk.y(1,i)
      mrk_start = mrk.pos(i);
      status = 1;
    end
   case 1   % session started, no pause, wait for end or pause start
    if mrk.y(2,i)
      mr = cat(1,mr,[mrk_start,mrk.pos(i)]);
      status = 0;
    end
    if mrk.y(3,i)
      mr = cat(1,mr,[mrk_start,mrk.pos(i)]);
      status = 2;
    end
   case 2   %paused wait for pause end
    if mrk.y(4,i)
      mrk_start = mrk.pos(i);
      status = 1;
    end
    if mrk.y(2,i)
      status = 0;
    end
  end
end

if status==1,
  mr = cat(1, mr, [mrk_start,mrk.pos(end)]);
  warning('last active phase has no end marker (took last marker instead)');
end
