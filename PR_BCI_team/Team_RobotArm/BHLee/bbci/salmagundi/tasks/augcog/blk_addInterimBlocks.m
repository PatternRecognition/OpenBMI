function blk= blk_addInterimBlocks(blk, className)
%blk= blk_addInterimBlocks(blk)
%
% side effect: calls blk_sortChronologically

if ~exist('className','var'),
  className= 'interim';
end

blk= blk_sortChronologically(blk);

interim= [];
for bb= 1:size(blk.ival,2)-1,
  new_iv= [blk.ival(2,bb)+1; blk.ival(1,bb+1)-1];
  if diff(new_iv)<-1,
    warning('overlapping blocks');
    continue;
  elseif diff(new_iv)==0,
    continue;
  end
  interim= cat(2, interim, new_iv);
end

if isempty(interim),
  warning('no interim blocks could be added');
  return;
end

blk.ival= cat(2, blk.ival, interim);
blk.y= [[blk.y, zeros(size(blk.y,1),size(interim,2))];
        [zeros(1,size(blk.y,2)), ones(1,size(interim,2))]];
blk.className= cat(2, blk.className, {className});

if isfield(blk, 'pos'),
  blk.pos= cat(2, blk.pos, interim(1,:));
end

blk= blk_sortChronologically(blk);
