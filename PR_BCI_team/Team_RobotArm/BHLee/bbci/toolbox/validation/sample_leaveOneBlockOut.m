function [divTr, divTe]= sample_leaveOneBlockOut(label, block_idx)
%[divTr, divTe]= sample_leaveOneBlockOut(label, block_idx)
%
% IN  label   - class labels, array of size [nClasses nSamples]
%               where row r indicates membership of class #r.
%               (0: no member, 1: member)
%     block_idx - [1 nSamples]-sized array of indices which defines
%               to which block each sample belongs. This can e.g., be
%               mrk.blkno, when mrk is obtained from mrk_evenlyInBlocks.
% 
% OUT divTr   - divTr{1}{n}: holds the training set for the case when
%               block with number n is th hold-out set.
%     divTe   - analogue to divTr, for the test set


divTr= cell(1, 1);
divTe= cell(1, 1);
idx_list= unique(block_idx);
for nn= 1:length(idx_list),
  divTe{1}(nn)= {find(block_idx==idx_list(nn))};
  divTr{1}(nn)= {find(block_idx~=idx_list(nn))};
end
