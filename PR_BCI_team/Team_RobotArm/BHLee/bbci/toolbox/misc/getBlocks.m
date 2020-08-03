function cnt = getBlocks(cnt,mr,flag);
%GETBLOCKS gives specific intervals out of cnt back.
%
% usage:
%   cnt = getBlocks(cnt,mr,<flag=false>); 
%
% input:
%   cnt     a usual cnt structure
%   mr      a nx2 matrices regarding the position in cnt of intervals
%   flag    divide blocks?? If true, cnt.x is a cell array of different block, if false the concatenated eeg is given back.
%
% output:
%   cnt     the modified cnt
%
% GUIDO DORNHEGE, 19/03/2004

if ~exist('flag','var') | isempty(flag)
  flag = false;
end

if flag
  dat = cnt.x;
  cnt.x = cell(1,size(mr,1));
  for i = 1:size(mr,1);
    cnt.x{i} = dat(mr(i,1):mr(i,2),:);
  end
else
  pos = [ ];
  for i = 1:size(mr,1);
    pos = [pos,mr(i,1):mr(i,2)];
  end
  cnt.x = cnt.x(pos,:);
end

  