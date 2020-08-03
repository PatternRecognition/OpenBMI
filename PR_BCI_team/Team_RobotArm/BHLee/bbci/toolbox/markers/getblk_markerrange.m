function blk= getblk_markerrange(mrk, mrk_orig, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'border', [0 0]);

is= strmatch('New Segment', mrk_orig.type);
nIvals= length(is);
for ii= 1:nIvals,
  seg(ii)= mrk_orig.pos(is(ii));
end
seg(length(is)+1)= inf;

blk= struct('fs',mrk.fs);
blk.ival= zeros(2, nIvals);
for ii= 1:nIvals,
  mrk_start= min(find(mrk.pos>seg(ii)));
  mrk_end= max(find(mrk.pos<seg(ii+1)));
  blk.ival(:,ii)= mrk.pos([mrk_start mrk_end]) + opt.border/1000*blk.fs;
end
