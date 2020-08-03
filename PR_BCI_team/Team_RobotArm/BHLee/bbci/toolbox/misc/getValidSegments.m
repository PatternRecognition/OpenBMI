function blk= getValidSegments(cnt, varargin)
%GETVALIDAREAS gives areas regarding mrk.pos back where the
% subject is not in a break
%
% usage:
%    area = getValidSegments(cnt, <opt>)
%    area = getValidSegments(file, <opt>)
%
% input:
%    cnt   struct of continuous EEG data
%    file  filename to load with readMarkerTable
%    opt
%     .session_start_marker:  an array of session_start_markers (252)
%     .session_end_marker:  an array of session_end_markers (253)
%     .pause_start_marker:  an array of pause_start_markers (249)
%     .pause_end_marker:  an array of pause_end_markers     (250)
%
% output:
%    blk   block segment struct with fields
%     .ival  an nx2 array with intervals
%     .fs    sampling rate
%
% GUIDO DORNHEGE; 19/03/2004

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'session_start_marker', 252, ...
                  'session_end_marker', 253, ...
                  'pause_start_marker', 249, ...
                  'pause_end_marker', 250, ...
                  'warning', 'on');

if ischar(cnt),
  cnt.file= cnt;
end
if ~isfield(cnt, 'fs'),
  if ~strcmp(opt.warning,'off'),
    warning('100Hz sampling rate assumed');
  end
  cnt.fs= 100;
end

if iscell(cnt.file),
  blk= getValidSegments(cnt.file{1}, opt, 'warning','off');
  blk_ival= blk.ival;
  for qq= 2:length(cnt.file),
    bk= getValidSegments(cnt.file{qq}, opt, 'warning','off');
    blk_ival= cat(1, blk_ival, bk.ival + cnt.T(qq-1));
  end
  blk.ival= blk_ival;
  return;
end

mrk= readMarkerTable(cnt.file, cnt.fs);

classDef = {opt.session_start_marker, opt.session_end_marker, ...
            opt.pause_start_marker, opt.pause_end_marker; ...
            'session start','session end','pause start','pause end'};

mrk = makeClassMarkers(mrk,classDef,0,0);

blk_ival= [];
status= 0;

for i= 1:size(mrk.y,2),
  switch status
   case 0   % no session started, wait for marker 1
    if mrk.y(1,i)
      mrk_start = mrk.pos(i);
      status = 1;
    end
   case 1   % session started, no pause, wait for end or pause start
    if mrk.y(2,i)
      blk_ival = cat(1, blk_ival, [mrk_start,mrk.pos(i)]);
      status = 0;
    end
    if mrk.y(3,i)
      blk_ival = cat(1, blk_ival, [mrk_start,mrk.pos(i)]);
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
  blk_ival = cat(1, blk_ival, [mrk_start,mrk.pos(end)]);
  warning('last active phase has no end marker (took last marker instead)');
end

blk= struct('ival',blk_ival, 'fs',mrk.fs);
