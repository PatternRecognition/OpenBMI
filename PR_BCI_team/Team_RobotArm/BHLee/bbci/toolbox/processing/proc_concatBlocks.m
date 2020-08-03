function [cnt, blk, mrk_new]= proc_concatBlocks(cnt, blk, mrk);
%PROC_CONCATBLOCKS extracts and concatenates specified blocks (intervals)
% of time series.
%
% usage:
%   [cnt, blk, mrk] = getBlocks(cnt, blk, <mrk>); 
%
% input:
%   cnt     a usual cnt structure
%   blk     block marker structure (fields fs and ival)
%
% output:
%   cnt     cnt structure consisting of the concatenated blocks
%   blk     block marker structure with condensed intervals
%   mrk     marker structure with condensed positions

% GUIDO DORNHEGE, 19/03/2004
% changed bb

if cnt.fs~=blk.fs,
  error('inconsistent sampling rates');
end

mrk_new= [];
idx= [];
bb= 0;
for ii= 1:size(blk.ival,2);
  blk_idx= blk.ival(1,ii):blk.ival(2,ii);
  idx= cat(2, idx, blk_idx);
  ival(1,ii)= bb+1;
  bb= bb + length(blk_idx);
  ival(2,ii)= bb;
  if exist('mrk','var'),
    if ii==1,
      offset= blk.ival(1,ii)-1;
    else
      offset= offset + blk.ival(1,ii) - blk.ival(2,ii-1) - 1;
    end
    iInBlock= find(mrk.pos>=blk.ival(1,ii) & mrk.pos<=blk.ival(2,ii));
    mk= mrk_selectEvents(mrk, iInBlock);
    mk.pos= mk.pos - offset;
    mrk_new= mrk_mergeMarkers(mrk_new, mk);
  end
end

cnt.x = cnt.x(idx,:);
blk.ival= ival;
