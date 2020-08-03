function [cnt, Mrk]= proc_appendCnt(cnt, cnt2, mrk, mrk2)

if cnt.fs~=cnt2.fs,
  error('mismatch in cnt sampling rates');
end

if exist('mrk','var'),
  if ~exist('mrk2','var'),
    error('you have to provide two marker structures');
  end
  if mrk.fs~=mrk2.fs,
    error('mismatch in mrk sampling rates');
  end
  if mrk.fs~=cnt.fs,
    error('mismatch between cnt and mrk sampling rates');
  end
end

if ~isequal(cnt.clab, cnt2.clab),
  sub= intersect(cnt.clab, cnt2.clab);
  if isempty(sub),
    error('data sets have disjoint channels');
  else
    msg= sprintf('mismatch in channels, using common subset (%d channels)', ...
                 length(sub));
    warning(msg);
  end
  cnt= proc_selectChannels(cnt);
  cnt2= proc_selectChannels(cnt2);
end

T= size(cnt.x, 1);
cnt.x= cat(1, cnt.x, cnt2.x);
if ~strcmp(cnt.title, cnt2.title),
  cnt.title= [cnt.title ' et al'];
end

if exist('mrk','var'),
  mrk2.pos= mrk2.pos + T;
  Mrk= mrk_mergeMarkers(mrk, mrk2);
end
