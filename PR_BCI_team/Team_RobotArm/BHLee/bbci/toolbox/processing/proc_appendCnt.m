function [cnt, mrk]= proc_appendCnt(cnt, cnt2, mrk, mrk2, varargin)
%PROC_APPENDCNT - Append Continuous EEG Data
%
%Synopsis:
% CNT= proc_appendCnt(CNT1, CNT2);
% CNT= proc_appendCnt({CNT1, ...});
% [CNT, MRK]= proc_appendCnt(CNT1, CNT2, MRK1, MRK2);
% [CNT, MRK]= proc_appendCnt({CNT1, ...}, {MRK1, ...});

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'channelwise', 0);

if iscell(cnt),
  %% iterative concatenation: not very effective
  if nargin>2,
    error('if the first argument is a cell: max 2 arguments');
  end
  cnt_cell= cnt;
  if nargin>1,
    mrk_cell= cnt2;
    if ~iscell(mrk_cell) | length(mrk_cell)~=length(cnt_cell),
      error('cell arrays for CNT and MRK do not match');
    end
  else
    mrk_cell= repmat([], 1, length(cnt_cell));
  end
  cnt= [];
  mrk= [];
  for cc= 1:length(cnt_cell),
    [cnt, mrk]= proc_appendCnt(cnt, cnt_cell{cc}, mrk, mrk_cell{cc});
  end
  return;
end

if exist('mrk','var'),
  if ~exist('mrk2','var'),
    error('you have to provide either no or two marker structures');
  end
end

if isempty(cnt),
  cnt= cnt2;
  if exist('mrk','var'),
    mrk= mrk2;
  end
  return;
end

if cnt.fs~=cnt2.fs,
  error('mismatch in cnt sampling rates');
end

if exist('mrk','var'),
  if ~isempty(mrk) & mrk(1).fs~=mrk2(1).fs,
    error('mismatch in mrk sampling rates');
  end
  if ~isempty(mrk) & mrk(1).fs~=cnt.fs,
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
  cnt= proc_selectChannels(cnt, sub);
  cnt2= proc_selectChannels(cnt2, sub);
end

T= size(cnt.x, 1);
C= size(cnt.x,2);
if opt.channelwise
  T2= size(cnt2.x, 1);
  for ic= 1:C
    cnt.x(1:T+T2,ic) = cat(1,cnt.x(1:T,ic),cnt2.x(:,ic));
  end
else
  cnt.x= cat(1, cnt.x, cnt2.x);
end
if ~strcmp(cnt.title, cnt2.title),
  cnt.title= [cnt.title ' et al'];
end
if isfield(cnt, 'file') & isfield(cnt2, 'file'),
  if ~iscell(cnt.file), cnt.file= {cnt.file}; end
  if ~iscell(cnt2.file), cnt2.file= {cnt2.file}; end
  cnt.file= cat(2, cnt.file, cnt2.file);
end

if exist('mrk','var') & ~isempty(mrk),
  if isfield(mrk(1),'time') & ~isfield(mrk2(1),'time')
    error('appending dissimilar structs for mrk');
  elseif length(mrk2)>1 || ~iscell(mrk2.type),
    %% mrk2 has format 'StructArray' (see eegfile_readBVmarkers)
    for i = 1:length(mrk2)
      mrk2(i).pos = mrk2(i).pos+T;
    end
    mrk = cat(1,mrk,mrk2);
  else
    mrk2.pos= mrk2.pos + T;
    mrk= mrk_mergeMarkers(mrk, mrk2);
  end
end
