function [tt, seg_breaks]= getTimeOfMarkers(mrk, seg)

margin= 10;

if ~isfield(mrk, 'T'),
  error('mrk structure needs to have a field T');
end

nSeg= length(mrk.T);
T= [0, cumsum(mrk.T)];
tt= zeros(1, length(mrk.pos));
seg_pauses= diff(seg)' - mrk.T(1:end-1)/mrk.fs/60;
for ii= 1:nSeg,
  idx= find(mrk.pos>T(ii) & mrk.pos<=T(ii+1));
  tt(idx)= mrk.pos(idx)/mrk.fs/60 + sum(seg_pauses(1:ii-1));
end

if nargout>1,
  seg_breaks= [seg(1:end-1)+mrk.T(1:end-1)'/mrk.fs/60, seg(2:end)];
  % correct end of segment to 'margin' sec. after last marker within
  % that segment
  for ii= 1:nSeg-1, 
    last_marker= max(find(tt<seg_breaks(ii,1)));
    seg_breaks(ii,1)= tt(last_marker) + margin/60;
  end
end
