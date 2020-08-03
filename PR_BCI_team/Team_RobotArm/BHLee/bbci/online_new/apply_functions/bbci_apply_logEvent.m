function bbci_apply_logEvent(data, bbci, icontrol)
%BBCI_APPLY_LOGEVENT - Log events to file or screen
%
%Synopsis:
%  bbci_apply_logEvent(DATA, BBCI)
%
%Arguments:
%  DATA - Structure of bbci_apply which holds all current data
%  BBCI - Structure of bbci_apply which specifies processing and 
%      classificaiton, type 'help bbci_apply_structures' for detailed
%      information about the fields of this structure.

% 03-2011 Benjamin Blankertz


persistent lastcall

if isequal(bbci.log.output, 0) || isequal(bbci.log.output, 'none'),
  return;
end

if ~bbci.log.classifier && isempty(data.control(icontrol).packet),
  return;
end

str= '';

if bbci.log.markers && icontrol==1,
  thiscall= data.marker.current_time;
  if isempty(lastcall) | thiscall<lastcall,
    check_ival= [0 thiscall];
  else
    check_ival= [lastcall thiscall];
  end
  lastcall= thiscall;
  new_events= bbci_apply_queryMarker(data.marker, check_ival);
  for k= 1:length(new_events),
    event= new_events(k);
    str= [str '# Marker: ' ...
          sprintf(bbci.log.time_fmt, event.time/1000), ' | '];
    if ischar(event.desc),
      str= [str, sprintf('M(%s)\n', event.desc)];
    else
      str= [str, sprintf('M(%d)\n', event.desc)];
    end
  end
end

if bbci.log.clock,
  str= [str, datestr(now, 'HH:MM:SS.FFF'), ' | '];
end

time_sec= data.source(icontrol).sample_no/data.source(icontrol).fs;
str= [str, sprintf(bbci.log.time_fmt, time_sec), ' | '];

if length(bbci.control)>1,
  str= [str, sprintf('CTRL%d | ', icontrol)];
end

% Log marker and event time, only if event is triggered by a
% marker condition
if ~isempty(bbci.control(icontrol).condition) && ...
      ~isempty(bbci.control(icontrol).condition.marker),
  if isempty(data.event.desc),
    str= [str, ': | '];
  else
    if ischar(data.event.desc),
      str= [str, sprintf('M(%s) | ', data.event.desc)];
    else
      str= [str, sprintf('M(%d) | ', data.event.desc)];
    end
  end
  str= [str, sprintf(bbci.log.time_fmt, data.event.time/1000), ' | '];
end

if bbci.log.classifier,
  str= [str, '['];
  for k= bbci.control(icontrol).classifier,
    if k~= bbci.control(icontrol).classifier(1),
      str= [str, ' '];
    end
    str= [str, vec2str(data.classifier(k).x,'%.13f',',')];
  end
  str= [str, '] | '];
end

packet= data.control(icontrol).packet;
str= [str, '{'];
for k= 1:length(packet)/2,
  if k>1,
    str= [str, ', '];
  end
  str= [str, packet{2*k-1}, '=', toString(packet{2*k})];
end
str= [str, '}\n'];

for k= 1:length(data.log.fid),
  fprintf(data.log.fid(k), str);
end
