function blk= blk_sortChronologically(blk)
%blk= blk_sortChronologically(blk)

[so,si]= sort(blk.ival(1,:));
blk.ival= blk.ival(:,si);
blk.y= blk.y(:,si);
if isfield(blk, 'pos'),
  blk.pos= blk.pos(si);
end
