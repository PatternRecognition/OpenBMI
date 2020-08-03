function blk= blk_sortChronologically(blk)

[so,si]= sort(blk.ival(1,:));
blk.ival= blk.ival(:,si);
if isfield(blk, 'y'),
  blk.y= blk.y(:,si);
end
