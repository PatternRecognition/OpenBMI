function epo = setBlockMarkers(epo,block_marker);
%SET DIVTR and DIVTE in epo to avoid block effects.
%
% usage:
%      epo = setBlockMarkers(epo,block_marker);
% 
% input:
%      epo   -  a usual epo structure
%      block_marker - a usual logical array describing the block dependencies
%
% output:
%      epo   - with field divTr and divTe

if ~exist('block_marker','var') | isempty(block_marker)
  return;
end

ind = find(sum(block_marker,2)>0);
block_marker = block_marker(ind,:);

epo.divTr = cell(1,size(block_marker,1));
epo.divTe = cell(1,size(block_marker,1));


for i = 1:size(block_marker,1);
  epo.divTe{i}{1} = find(block_marker(i,:));
  epo.divTr{i}{1} = find(block_marker(i,:)==0);
end

  