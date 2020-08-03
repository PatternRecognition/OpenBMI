function mrk2 = mrk_setMarkers(mrk,spacing,flag);
%MRK_SETMARKERS sets markers into mrk in specified time steps
%
% usage:
%   mrk = mrk_setMarkers(mrk,<spacing=1000,1000>);
%
% description: starting from each point in mrk.pos a new marker (with
%   the same labelling) is set in specified steps defined by spacing(1)
%   and is stopped until the next marker in mrk.pos is reached. The
%   last such defined marker has at least spacing(2) place to the next
%   mrk.pos. 
%
% input:
%   mrk      a usual mrk structure
%   spacing  time step between two markers and minimum time step of the last to the next block
%
% output:
%   mrk      the updated mrk structure
%
% GUIDO DORNHEGE, 13/02/2004

if ~exist('spacing','var') | isempty(spacing)
  spacing = [1000,1000,0];
end

if ~exist('flag','var') | isempty(flag)
  flag = 0;
end

if isfield(mrk,'ende') & ~isfield(mrk,'end');
  mrk.end = mrk.ende;
end

if length(spacing)==1, spacing = [1 1]*spacing;end

if length(spacing)==2, spacing = [spacing,0];end

if isfield(mrk,'ival')
  pos = mrk.ival;
else
  pos = [mrk.pos; mrk.pos(2:end)-1,mrk.end];
end

spacing = spacing/1000*mrk.fs;

mr = copyStruct(mrk,'pos','y','toe','end');

for i = 1:size(pos,2)
  pp = pos(1,i)+spacing(3):spacing(1):pos(2,i)-spacing(2);
  mr.pos = pp;
  mr.y = repmat(mrk.y(:,i),[1,length(pp)]);
  if isfield(mrk,'toe')
    mr.toe = repmat({mrk.toe{i}},[1,length(pp)]);
  end
  if isfield(mrk,'indexedByEpochs')
    for fld = mrk.indexedByEpochs
      eval(sprintf('mr.%s = mrk.%s(:,i).*ones(size(mrk.%s,1),length(mr.pos));',fld{1},fld{1},fld{1}));
    end
  end
  
  if flag
    mr.bidx = i*ones(1,length(mr.pos));
    if isfield(mr,'indexedByEpochs')
      mr.indexedByEpochs = {mr.indexedByEpochs{:},'bidx'};
    else
      mr.indexedByEpochs = {'bidx'};
    end
  end
  if i == 1
    mrk2 = mr;
  else
    mrk2 = mrk_mergeMarkers(mrk2,mr);
  end
end

  
  