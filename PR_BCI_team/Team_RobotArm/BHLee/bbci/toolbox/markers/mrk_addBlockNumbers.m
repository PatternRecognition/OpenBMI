function mrk= mrk_addBlockNumbers(mrk, blk)
% MRK= mrk_addBlockNumbers(MRK, BLK)
%
% See also:
%   blk_segmentsFromMarkers, mrk_evenlyInBlocks

mrk.block_no= NaN*zeros(1, length(mrk.pos));
for bb= 1:size(blk.ival,2),
  idx= find(mrk.pos>=blk.ival(1,bb) & mrk.pos<=blk.ival(2,bb));
  mrk.block_no(idx)= bb;
end
mrk= mrk_addIndexedField(mrk, 'block_no');
