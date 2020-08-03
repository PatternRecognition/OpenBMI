function fv_block= proc_subtrialFeatures(fv, out, nClasses)

nBlocks= length(out)/nClasses;
fv_block.x= reshape(out, [nClasses nBlocks]);
fv_block.y= reshape([1 0]*fv.y, [nClasses nBlocks]);
fv_block.className= cprintf('class %d', 1:nClasses)';
fv_block.block_idx= fv.block_idx(1:nClasses:end);
fv_block.trial_idx= fv.trial_idx(1:nClasses:end);
fv_block.indexedByEpochs= {'block_idx', 'trial_idx'};
if isfield(fv, 'stimulus'),
  fv_block.stimulus= reshape(fv.stimulus, [nClasses nBlocks]);
  fv_block= mrk_addIndexedField(fv_block, 'stimulus');
end
if isfield(fv, 'mode'),
  fv_block.mode= fv.mode(:,1:nClasses:end);
  fv_block= mrk_addIndexedField(fv_block, 'mode');
end
if isfield(fv, 'level'),
  fv_block.level= fv.level(:,1:nClasses:end);
  fv_block= mrk_addIndexedField(fv_block, 'level');
end
