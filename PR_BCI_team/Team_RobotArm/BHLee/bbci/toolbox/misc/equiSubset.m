function subs= equiSubset(idx)
%subs= equiSubset(idx)
%
% idx is a cell array of indices. subs will be a cell array, where each
% cell is a subset of the corresponding idx cell, all of the same size.

nClasses= length(idx);
for ci= 1:nClasses,
  len(ci)= length(idx{ci});
end
N= min(len);
subs= cell(1,nClasses);
for ci= 1:nClasses,
  rp= randperm(len(ci));
  subs{ci}= idx{ci}(rp(1:N));
end

