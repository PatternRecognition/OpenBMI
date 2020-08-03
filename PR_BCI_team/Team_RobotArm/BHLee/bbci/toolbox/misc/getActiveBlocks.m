function blk = getActiveBlocks(mrk, varargin);
%GETACTIVEBLOCKS gives areas regarding mrk.pos back where the subject is not in a break
%
%Synopsis:
%  blk = getActiveBlocks(mrk, <opt>)
%
%Arguments:
%   mrk - BV marker structure (struct of arrays) such as return by
%          eegfile_readBVmarkers(..., 0) or
%         file name (in this case markers are loaded from that file)
%   opt:
%    fs: resample markers to this sampling rate
%    session_start_marker:  an array of session_start_markers (252)
%    session_end_marker:  an array of session_end_markers (253)
%    pause_start_marker:  an array of pause_start_markers (249)
%    pause_end_marker:  an array of pause_end_markers     (250)
%
%Returns:
%   blk      an nx2 array with intervals

% GUIDO DORNHEGE; 19/03/2004 (getActivation Areas)
% Benjamin Blankertz Oct 2006

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'session_start_marker', 252, ...
                  'session_end_marker', 253, ...
                  'pause_start_marker', 249, ...
                  'pause_end_marker', 250);


classDef = {opt.session_start_marker, opt.session_end_marker, ...
            opt.pause_start_marker, opt.pause_end_marker; ...
            'session start','session end','pause start','pause end'};

if isstruct(mrk),
  if isfield(mrk, 'className'),
    mrk= mrk_selectClasses(mrk, classDef(1,:));
  else
    mrk= mrk_defineClasses(mrk, classDef);
  end
else
  [mrk, fs]= eegfile_loadMatlab(mrk, 'vars',{'mrk_orig','fs_orig'});
  mrk= mrk_arrayOfStructs2structOfArrays(mrk, fs);
  mrk= mrk_resample(mrk, opt.fs);
  mrk= mrk_defineClasses(mrk, classDef);
end

blk = [];
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
      blk = cat(1,blk,[mrk_start,mrk.pos(i)]);
      status = 0;
    end
    if mrk.y(3,i)
      blk = cat(1,blk,[mrk_start,mrk.pos(i)]);
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
  blk = cat(1, blk, [mrk_start,mrk.pos(end)]);
  warning('last active phase has no end marker (took last marker instead)');
end

blk= struct('fs',mrk.fs, 'ival',blk');
